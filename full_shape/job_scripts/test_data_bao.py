"""
Script to run BAO fits.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python test_data_bao.py
```
"""
import argparse
import os
from pathlib import Path

import numpy as np

from full_shape import tools, setup_logging


setup_logging()


def run_fit(actions=('profile',), tracer='LRG1', data='data-dr2-v1.1', stats_dir=Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base'), fits_dir=Path(os.getenv('SCRATCH')) / 'fits'):
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
    from full_shape.tools import generate_likelihood_options_helper, fill_fiducial_options, get_likelihood
    template = 'bao'
    options = {}
    # use post-reconstruction correlation function
    options['likelihoods'] = [generate_likelihood_options_helper(stats=['recon_particle2_correlation'], tracer=tracer, version=data, stats_dir=stats_dir, emulator=template == 'direct')]
    for likelihood_options in options['likelihoods']:
        # rascalc = analytical covariance
        likelihood_options['covariance'] = {'source': 'rascalc', 'version': 'data-dr2-v1.1', 'stats_dir': stats_dir}
    options['cosmology'] = {'template': template, 'apmode': 'qisoqap'}
    options = fill_fiducial_options(options)
    # options contains all possible options; print(options) to look at its content
    get_fits_fn = functools.partial(tools.get_fits_fn, fits_dir=fits_dir)
    run_fit_from_options(actions, **options, get_fits_fn=get_fits_fn, cache_dir=None)
    likelihood = get_likelihood(likelihoods_options=options['likelihoods'],
                                cosmology_options=options['cosmology'], cache_dir=None)
    fn = get_fits_fn(kind='profiles', **options)
    from desilike.samples import Profiles
    profiles = Profiles.load(fn)
    # evaluate likelihood at dictionary of parameters
    likelihood(**profiles.bestfit.choice(input=True, index='argmax'))
    likelihood.likelihoods[0].observables[0].plot(fn=f'./_tests/plot_{tracer}.png')
    likelihood.likelihoods[0].observables[0].plot_bao(fn=f'./_tests/plot_bao_{tracer}.png')


if __name__ == '__main__':

    stats_dir = Path(f'/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    fits_dir = Path(os.getenv('SCRATCH')) / 'fits_bao'

    for tracer in ['LRG1', 'LRG2', 'LRG3', 'ELG1', 'ELG2', 'QSO'][:3]:
        run_fit(actions=['profile'], data='data-dr2-v1.1', tracer=tracer, stats_dir=stats_dir, fits_dir=fits_dir)