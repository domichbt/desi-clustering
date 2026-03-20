"""
Script to run fits with Abacus mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python validation_abacus_mocks.py
```
"""
import os
from pathlib import Path

import numpy as np

from full_shape import tools, setup_logging


setup_logging()


def run_fit(actions=('profile',), template='direct', version='abacus-2ndgen-complete', fits_dir=Path(os.getenv('SCRATCH')) / 'fits', stats=['mesh2_spectrum']):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    from pathlib import Path
    import functools
    from full_shape.tools import get_default_mpicomm
    mpicomm = get_default_mpicomm()
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(mpicomm.rank)
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    from full_shape import run_fit_from_options, setup_logging
    options = {}
    # You can pass region, version, covariance, ...
    #likelihoods_options = [generate_likelihood_options_helper(tracer=tracer, version=version) for tracer in ['LRG1', 'LRG2', 'LRG3', 'ELG2', 'QSO1']]
    options['likelihoods'] = [tools.generate_likelihood_options_helper(tracer=tracer, version=version) for tracer in ['LRG1']]
    # template = direct, shapefit
    options['cosmology'] = {'template': template}
    run_fit_from_options(actions, **options, get_fits_fn=functools.partial(tools.get_fits_fn, fits_dir=fits_dir), cache_dir='./_cache')


if __name__ == '__main__':

    fits_dir = Path(os.getenv('SCRATCH')) / 'fits_abacus_mocks'
    version = 'abacus-2ndgen-complete'
    run_fit(actions=['profile'], version=version, fits_dir=fits_dir)

