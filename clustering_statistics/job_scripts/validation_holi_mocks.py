"""
Script to create and spawn desipipe tasks to compute clustering measurements on HOLI mocks.
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 validation_holi_mocks.py
```
"""
import os
from pathlib import Path

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()


def run_stats(tracer='LRG', version='holi-v1-altmtl', weight='default-FKP', zranges=None, imocks=[451], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum']):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    import jax
    from jax import config
    config.update('jax_enable_x64', True)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, combine_stats_from_options, fill_fiducial_options

    setup_logging()
    cache = {}
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer)
    for imock in imocks:
        regions = ['NGC', 'SGC']
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), mesh2_spectrum={'cut': True, 'auw': True if 'altmtl' in version else None})
            options = fill_fiducial_options(options)
            for _tracer in options['catalog']:
                options['catalog'][_tracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=_tracer, nran=options['catalog'][_tracer]['nran'])}
            compute_stats_from_options(stats, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), cache=cache, **options)
    #jax.distributed.shutdown()


def postprocess_stats(tracer='LRG', version='holi-v1-altmtl', complete=False, imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', postprocess=['combine_regions'], **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    import functools
    zranges = tools.propose_fiducial('zranges', tracer)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, imock=imocks[0]), imocks=imocks, combine_regions={'stats': ['mesh2_spectrum', 'mesh3_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'window_mesh3_spectrum'][3:4]}, mesh2_spectrum={'cut': True}, window_mesh2_spectrum={'cut': True})
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)



if __name__ == '__main__':

    imocks = 0 + np.arange(1)

    stats_dir = Path(os.getenv('SCRATCH')) / 'covariance_holi_mocks'
    #stats_dir = Path("/global/cfs/projectdirs/desi/mocks/cai/mock-benchmark-dr2/summary_statistics/cutsky/")
    stats = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum'][:0]
    postprocess = ['combine_regions'][:1]

    version = 'holi-v3-altmtl'

    for tracer in ['LRG', 'ELG_LOPnotqso', ('LRG', 'ELG_LOPnotqso')]:
        #for weight in ['default_compntile', 'default']:
        #    run_stats(tracer, version='holi-v3-complete', weight=weight, imocks=imocks, stats_dir=stats_dir)
        weight = 'default-FKP'
        zrange = (0.8, 1.1)
        if any('window' in stat or 'covariance' in stat for stat in stats):
            imocks = [0]
        if stats:
            run_stats(tracer, version=version, weight=weight, imocks=imocks, zranges=[zrange], stats=stats, stats_dir=stats_dir)
        if postprocess:
            postprocess_stats(tracer=tracer, version=version, imocks=imocks, stats_dir=stats_dir, postprocess=postprocess)