"""
Script to run fits with Abacus mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python validation_abacus_mocks.py --dataset abacus-2ndgen-dr2-complete --tracers LRG1 --stats mesh2_spectrum mesh3_spectrum --todo profile sample
```
"""
import argparse
import os
from pathlib import Path

import numpy as np

from full_shape import tools, setup_logging


setup_logging()


def run_fit(actions=('profile',), template='direct', version='abacus-2ndgen-dr2-complete',
            covariance='holi-v1-altmtl',
            stats_dir=Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base'),
            fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits',
            stats=['mesh2_spectrum'], tracers=None):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    from pathlib import Path
    import functools
    from desilike.mpi import CurrentMPIComm
    mpicomm = CurrentMPIComm.get()
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(mpicomm.rank)
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    from full_shape import run_fit_from_options, setup_logging
    from full_shape.tools import get_likelihood
    from desilike.samples import Profiles
    options = {}
    # You can pass region, version, covariance, ...
    options['likelihoods'] = [tools.generate_likelihood_options_helper(stats=stats, tracer=tracer, version=version, covariance=covariance, stats_dir=stats_dir) for tracer in tracers]
    # template = direct, shapefit
    options['cosmology'] = {'template': template}
    options = tools.fill_fiducial_options(options)
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
                    plot_covariance = plot_covariance.at.observable.match(observable.data.clone(value=0. * observable.data.value()))
                    plot_observable = observable.__class__(
                        data=observable.data,
                        window=observable.window,
                        covariance=plot_covariance,
                        theory=observable.theory,
                    )
                    plot_observable()
                    plot_observable.plot(fn=plot_dir / f'plot_likelihood{ilikelihood}_observable{iobservable}.png')


if __name__ == '__main__':
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
    parser.add_argument('--tracers', action='extend', nargs='+', default=None,
                        help='Tracer(s) to fit. Pass one or more values after --tracers. Defaults to LRG1.')
    parser.add_argument('--fits_dir', type=str, default=None,
                        help='Base directory for fits. Defaults to $SCRATCH/fits_abacus_mocks or ./fits_abacus_mocks.')
    args = parser.parse_args()

    base_fits_dir = Path(args.fits_dir) if args.fits_dir is not None else Path(os.getenv('SCRATCH', '.')) / 'fits_abacus_mocks'
    fits_dir = base_fits_dir / args.dataset
    version = args.dataset
    covariance = 'holi-v3-altmtl'
    stats_dir = Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base')
    stats = args.stats
    tracers = args.tracers or ['LRG1']
    run_fit(actions=args.todo, version=version, covariance=covariance, stats_dir=stats_dir,
            fits_dir=fits_dir, stats=stats, tracers=tracers)
