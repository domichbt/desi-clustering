"""
Script to run fits with Abacus mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python validation_abacus_mocks.py --dataset abacus-2ndgen-dr2-complete --tracers LRG1 --stats mesh2_spectrum mesh3_spectrum --todo profile sample
srun -n 4 python validation_abacus_mocks.py --dataset abacus-2ndgen-dr2-complete --tracers LRG1 LRG2 LRG3 ELG2 --stats mesh2_spectrum mesh3_spectrum --theory_model folpsEFT --cosmo_params base_ns-fixed --todo sample --nchains 4
```
"""
import argparse
import os
from pathlib import Path

import numpy as np

from full_shape import tools, setup_logging


setup_logging()


THEORY_MODELS = ['folpsD', 'folpsEFT', 'reptvelocileptors']
COSMO_MODELS = ['base', 'base_ns-fixed']


def _validate_theory_model(stats, theory_model):
    if theory_model == 'reptvelocileptors' and 'mesh3_spectrum' in stats:
        raise ValueError('theory model reptvelocileptors is only supported with mesh2_spectrum')


def _build_likelihoods_options(stats, tracers, version, covariance, stats_dir, theory_model):
    _validate_theory_model(stats, theory_model)
    likelihoods = []
    for tracer in tracers:
        likelihood_options = tools.generate_likelihood_options_helper(
            stats=stats,
            tracer=tracer,
            version=version,
            covariance=covariance,
            stats_dir=stats_dir,
        )
        for observable_options in likelihood_options['observables']:
            observable_options.setdefault('theory', {})
            observable_options['theory']['model'] = theory_model
        likelihoods.append(likelihood_options)
    return likelihoods


def _build_run_options(stats, tracers, version, covariance, stats_dir, theory_model,
                       cosmo_model='base', template='direct', nchains=1):
    options = {}
    options['likelihoods'] = _build_likelihoods_options(
        stats=stats,
        tracers=tracers,
        version=version,
        covariance=covariance,
        stats_dir=stats_dir,
        theory_model=theory_model,
    )
    options['cosmology'] = {'template': template, 'model': cosmo_model}
    options['sampler'] = {'nchains': nchains}
    return tools.fill_fiducial_options(options)


def run_fit(actions=('profile',), template='direct', version='abacus-2ndgen-dr2-complete',
            covariance='holi-v1-altmtl',
            stats_dir=Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base'),
            fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits',
            stats=['mesh2_spectrum'], tracers=None, theory_model='folpsD',
            cosmo_model='base', nchains=1):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    from pathlib import Path
    import functools
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(mpicomm.rank)
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    from full_shape import run_fit_from_options, setup_logging
    from full_shape.tools import get_likelihood
    from desilike.samples import Profiles
    # You can pass region, version, covariance, ...
    options = _build_run_options(
        stats=stats,
        tracers=tracers,
        version=version,
        covariance=covariance,
        stats_dir=stats_dir,
        theory_model=theory_model,
        cosmo_model=cosmo_model,
        template=template,
        nchains=nchains,
    )
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir)
    run_fit_from_options(actions, **options, get_fits_fn=get_fits_fn, cache_dir='./_cache')
    if 'profile' in actions:
        likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                    cosmology_options=options['cosmology'],
                                    cache_dir='./_cache')
        profiles = Profiles.load(get_fits_fn(kind='profiles', **options))
        likelihood(**profiles.bestfit.choice(input=True, index='argmax'))
        if mpicomm.rank == 0:
            plot_dir = get_fits_fn(kind='profiles', **options).parent
            for ilikelihood, sublikelihood in enumerate(likelihood.likelihoods):
                for iobservable, observable in enumerate(sublikelihood.observables):
                    plot_covariance = sublikelihood.covariance.at.observable.get(observables=observable.name)
                    plot_covariance = plot_covariance.at.observable.match(observable.data)
                    observable.covariance = plot_covariance
                    observable.plot(fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.png')


def _get_parser():
    datasets = ['abacus-2ndgen-dr2-altmtl', 'abacus-2ndgen-dr2-complete', 'abacus-hf-dr2-v2-altmtl']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=datasets, default='abacus-2ndgen-dr2-complete',
                        help='Dataset to fit. Defaults to abacus-2ndgen-dr2-complete.')
    parser.add_argument('--todo', type=str, nargs='*', default=['profile'],
                        choices=['build', 'profile', 'sample'],
                        help='Run build, profile, and / or sample. Defaults to profile.')
    parser.add_argument('--stats', type=str, nargs='*', default=['mesh2_spectrum'],
                        choices=['mesh2_spectrum', 'mesh3_spectrum'],
                        help='Statistics to fit. Defaults to mesh2_spectrum.')
    parser.add_argument('--theory_model', type=str, default='folpsD',
                        choices=THEORY_MODELS,
                        help='Theory model to fit. Defaults to folpsD.')
    parser.add_argument('--cosmo_params', type=str, default='base',
                        choices=COSMO_MODELS,
                        help='Cosmology parameter setup to fit. base varies h, omega_cdm, omega_b, logA, n_s; '
                             'base_ns-fixed varies h, omega_cdm, omega_b, logA. Defaults to base.')
    parser.add_argument('--tracers', action='extend', nargs='+', default=None,
                        help='Tracer(s) to fit. Pass one or more values after --tracers. Defaults to LRG1.')
    parser.add_argument('--fits_dir', type=str, default=None,
                        help='Base directory for fits. Defaults to $SCRATCH/fits_abacus_mocks or ./fits_abacus_mocks.')
    parser.add_argument('--nchains', type=int, default=1,
                        help='Number of MCMC chains to run with desilike. Defaults to 1.')
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()

    base_fits_dir = Path(args.fits_dir) if args.fits_dir is not None else Path(os.getenv('SCRATCH', '.')) / 'fits_abacus_mocks'
    fits_dir = base_fits_dir / args.dataset
    version = args.dataset
    covariance = 'holi-v3-altmtl'
    stats_dir = Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base')
    stats = args.stats
    tracers = args.tracers or ['LRG1']
    _validate_theory_model(stats, args.theory_model)
    run_fit(actions=args.todo, version=version, covariance=covariance, stats_dir=stats_dir,
            fits_dir=fits_dir, stats=stats, tracers=tracers, theory_model=args.theory_model,
            cosmo_model=args.cosmo_params, nchains=args.nchains)
