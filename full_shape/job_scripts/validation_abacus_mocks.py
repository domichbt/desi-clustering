"""
Script to run fits with Abacus mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python validation_abacus_mocks.py --dataset abacus-2ndgen-dr2-complete --tracer LRG1
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
    options = {}
    # You can pass region, version, covariance, ...
    options['likelihoods'] = [tools.generate_likelihood_options_helper(tracer=tracer, version=version, covariance=covariance, stats_dir=stats_dir) for tracer in tracers]
    # template = direct, shapefit
    options['cosmology'] = {'template': template}
    run_fit_from_options(actions, **options, get_fits_fn=functools.partial(tools.get_fits_fn, fits_dir=fits_dir), cache_dir='./_cache')


if __name__ == '__main__':
    datasets = ['abacus-2ndgen-dr2-altmtl', 'abacus-2ndgen-dr2-complete', 'abacus-hf-dr2-v2-altmtl']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=datasets, default='abacus-2ndgen-dr2-complete',
                        help='Dataset to fit. Defaults to abacus-2ndgen-dr2-complete.')
    parser.add_argument('--tracer', action='extend', nargs='+', dest='tracers', default=None,
                        help='Tracer(s) to fit. Pass one or more values after --tracer. Defaults to LRG1.')
    parser.add_argument('--fits_dir', type=str, default=None,
                        help='Base directory for fits. Defaults to $SCRATCH/fits_abacus_mocks or ./fits_abacus_mocks.')
    args = parser.parse_args()

    base_fits_dir = Path(args.fits_dir) if args.fits_dir is not None else Path(os.getenv('SCRATCH', '.')) / 'fits_abacus_mocks'
    fits_dir = base_fits_dir / args.dataset
    version = args.dataset
    covariance = 'holi-v3-altmtl'
    stats_dir = Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base')
    tracers = args.tracers or ['LRG1']
    run_fit(actions=['profile'], version=version, covariance=covariance, stats_dir=stats_dir,
            fits_dir=fits_dir, tracers=tracers)
