"""
Script to create and spawn desipipe tasks to compute clustering measurements on glam mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
srun -n 4 validation_glam-uchuu_mocks.py
```
"""
import os
from pathlib import Path
import functools
import sys

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()


def run_stats(tracer='LRG', version='glam-uchuu-v1-altmtl', weight='default-FKP', imocks=[451], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], onthefly=None):
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

    if 'complete' in version:
        cat_dir = Path(os.getenv('SCRATCH')) / 'clustering_catalogs' / f'glam-uchuu-v1-altmtl_complete'

        def get_catalog_fn(imock=0, **kwargs):
            return tools.get_catalog_fn(cat_dir=cat_dir / f'complete{imock:d}', imock=imock, **kwargs)

    else:
        get_catalog_fn = tools.get_catalog_fn

    setup_logging()
    cache = {}
    zranges = tools.propose_fiducial('zranges', tracer)
    for imock in imocks:
        regions = ['NGC', 'SGC'][:1]
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), mesh2_spectrum={'cut': True, 'auw': True if 'altmtl' in version else None})
            if onthefly == 'complete':
                options['catalog']['complete'] = {}
                get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, extra='complete')
            else:
                get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
            options = fill_fiducial_options(options)
            if 'complete' not in version:
                for tracer in options['catalog']:
                    options['catalog'][tracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=options['catalog'][tracer]['nran'])}
            compute_stats_from_options(stats, get_stats_fn=get_stats_fn, get_catalog_fn=get_catalog_fn, cache=cache, **options)
    #jax.distributed.shutdown()


def postprocess_stats(tracer='LRG', version='glam-uchuu-v1-altmtl', weight='default', imocks=[0], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', postprocess=['combine_regions'], **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    zranges = tools.propose_fiducial('zranges', tracer)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'mesh3_spectrum', 'window_mesh3_spectrum'][1:3]})
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)


def plot_density(imock=[0], tracer='LRG', zranges=None, version='glam-uchuu-v1-altmtl', weight='default', plots_dir=Path('./_plots'), nside=8, get_catalog_fn=tools.get_catalog_fn):
    from clustering_statistics.density_tools import plot_density_projections
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer)
    for region in ['NGC', 'SGC']:
        for zrange in zranges:
            zstep = 0.01
            edges = {'Z': np.arange(zrange[0], zrange[1] + zstep, zstep),
                     'RA': np.linspace(0., 360., 361),
                     'DEC': np.linspace(-90., 90., 181)}
            catalog = dict(version=version, tracer=tracer, zrange=zrange, region=region, weight=weight, nran=5)
            catalog['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog['nran'])}
            plot_density_projections(get_catalog_fn=get_catalog_fn, divide_randoms=True, catalog=catalog,
                                     imock=imock, edges=edges, fn=plots_dir / f'density_fluctuations_{version}_weight-{weight}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png', nside=nside)
            plot_density_projections(get_catalog_fn=get_catalog_fn, divide_randoms=False, catalog=catalog,
                                     imock=imock, edges=edges, fn=plots_dir / f'density_{version}_weight-{weight}_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}.png', nside=nside)


if __name__ == '__main__':

    stats_dir = Path(os.getenv('SCRATCH')) / 'glam-uchuu_mocks_validation'
    imocks = 100 + np.arange(50)[:3]

    todo = ['stats']
    stats = ['mesh2_spectrum', 'mesh3_spectrum_sugiyama', 'mesh3_spectrum_scoccimarro', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum'][:1]
    postprocess = ['combine_regions'][:0]
    weight = 'default-FKP'
    version = 'glam-uchuu-v1-altmtl'
    #version = 'glam-uchuu-v1-complete'
    onthefly = 'complete'
    #onthefly = None

    if 'stats' in todo:
        tracers = ['LRG', 'ELG_LOPnotqso', 'QSO']
        for tracer in tracers:
            if any('window' in stat or 'covariance' in stat for stat in stats):
                imocks = [100]
            if stats:
                run_stats(tracer, version=version, weight=weight, stats=stats, onthefly=onthefly, stats_dir=stats_dir, imocks=imocks)
            if postprocess:
                postprocess_stats(tracer, version=version, weight=weight, imocks=imocks, stats_dir=stats_dir, postprocess=postprocess)

    if 'density' in todo:
        tracers = ['LRG', 'ELG_LOPnotqso', 'QSO']

        for tracer in tracers:
            plot_density(imock=imocks, tracer=tracer, version=version, weight='default')
