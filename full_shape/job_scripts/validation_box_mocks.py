"""
Script to run fits with Abacus-2ndgen cubic box mocks.
Data vector = mean of all available mock realizations (abacus-2ndgen).
Covariance = estimated from ezmock-dr1 mocks.

To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python validation_box_mocks.py
```
"""
import os
import functools
from pathlib import Path

from full_shape import tools, setup_logging


setup_logging()

BOX_STATS_DIR = Path('/dvs_ro/cfs/cdirs/desicollab/mocks/cai/LSS/DA2/mocks/desipipe/box')


def run_fit(actions=('profile',), template='direct',
            fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits_box_mocks',
            stats=['mesh2_spectrum']):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained and cannot rely on imports from the outer scope.
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
    from clustering_statistics import box_tools
    from full_shape.tools import generate_box_likelihood_options_helper, get_fits_fn
    BOX_STATS_DIR = Path('/dvs_ro/cfs/cdirs/desicollab/mocks/cai/LSS/DA2/mocks/desipipe/box')
    options = {}
    options['likelihoods'] = [
        generate_box_likelihood_options_helper(
            tracer='LRG', zsnap=0.800, stats=stats,
            stats_dir=BOX_STATS_DIR,
        )
    ]
    options['cosmology'] = {'template': template}
    run_fit_from_options(
        actions, **options,
        get_stats_fn=box_tools.get_box_stats_fn,
        get_fits_fn=functools.partial(get_fits_fn, fits_dir=fits_dir),
        cache_dir='./_cache',
    )


if __name__ == '__main__':
    fits_dir = Path(os.getenv('SCRATCH', '.')) / 'fits_box_mocks'
    run_fit(actions=['profile'], fits_dir=fits_dir)
