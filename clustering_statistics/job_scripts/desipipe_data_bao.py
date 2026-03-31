"""
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python job_scripts/desipipe_data_bao.py
desipipe tasks -q data_bao  # check the list of tasks
desipipe spawn -q data_bao --spawn  # spawn the jobs
desipipe queues -q data_bao  # check the queue
Or directly if mode = 'interactive':
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
srun -n 4 python job_scripts/desipipe_data_bao.py
"""

import os, sys
import itertools
import numpy as np
import functools
from pathlib import Path
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools
setup_logging()

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD

queue = Queue('data_bao')
queue.clear(kill=False)

output, error = './slurm_outputs/data_bao/slurm-%j.out', './slurm_outputs/data_bao/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))


def run_stats(version='data-dr2-v1.1', tracer='LRG', weight_type='weight-FKP', zranges=None, stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], ibatch=None, **kwargs):
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
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options
    setup_logging()
    cache = {}
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer)
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    for region in ['NGC', 'SGC']:
        battrs = None #dict(s=np.linspace(0., 150., 151), mu=(np.linspace(-1., 1., 201), 'midpoint'))
        particle2_correlation = {'battrs': battrs, 'jackknife': {'nsplits': 60}}
        options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight_type=weight_type), mesh2_spectrum={}, window_mesh2_spectrum={}, particle2_correlation=particle2_correlation, recon_particle2_correlation=particle2_correlation, window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
        options = fill_fiducial_options(options)
        compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, **options)


def postprocess_stats(version='data-dr2-v1.1', tracer='LRG', weight='default-FKP', stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', postprocess=['combine_regions'], **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    zranges = tools.propose_fiducial('zranges', tracer)
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    particle2_correlation = {'jackknife': {'nsplits': 60}}
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight), combine_regions={'stats': ['mesh2_spectrum', 'mesh3_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'window_mesh3_spectrum', 'recon_particle2_correlation']}, particle2_correlation=particle2_correlation, recon_particle2_correlation=particle2_correlation)
    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)


if __name__ == '__main__':

    mode = 'interactive'
    stats = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'recon_particle2_correlation'][:0]
    postprocess = ['combine_regions'][:1]

    stats_dir = Path(f'/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    #version = 'data-dr2-v1.1'
    version = 'data-dr1-v1.5'

    for tracer in ['BGS', 'LRG', 'ELG', 'QSO'][1:2]:
        tracer = tools.get_full_tracer(tracer, version=version)
        zranges = tools.propose_fiducial('zranges', tracer)[:1]

        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)
    
        if stats:
            get_run_stats()(version=version, tracer=tracer, zranges=zranges, stats_dir=stats_dir, stats=stats)
        if postprocess:
            postprocess_stats(version=version, tracer=tracer, zranges=zranges, stats_dir=stats_dir, postprocess=postprocess)
