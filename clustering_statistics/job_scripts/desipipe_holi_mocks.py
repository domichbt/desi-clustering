"""
Script to create and spawn desipipe tasks to compute clustering measurements on HOLI mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 01:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=$HOME/cai-dr2-clustering-products/:$PYTHONPATH
python desipipe_holi_mocks.py         # create the list of tasks
desipipe tasks -q holi_mocks          # check the list of tasks
desipipe spawn -q holi_mocks --spawn  # spawn the jobs
desipipe queues -q holi_mocks         # check the queue
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

queue = Queue('holi_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/holi_mocks/slurm-%j.out', 'slurm_outputs/holi_mocks/slurm-%j.err'
kwargs = {}
environ = Environment('nersc-cosmodesi')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))


def run_stats(tracer='LRG', project='', version='holi-v1-altmtl', complete=False, imocks=[201], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], analysis='full_shape', ibatch=None, **kwargs):
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
    zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    for imock in imocks:
        regions = ['NGC', 'SGC']
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, imock=imock), mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True}, window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
            if complete:
                options['catalog']['complete'] = {}
                get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project, extra='complete')
            else:
                get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project)
            #    options['catalog']['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', **(options['catalog'] | dict(region='ALL')))}
            #    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project, extra='reshuffle')
            options = fill_fiducial_options(options, analysis=analysis)
            #for tracer in options['catalog']:
            #    options['catalog'][tracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=options['catalog'][tracer]['nran'])}
            compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, **options)


def postprocess_stats(tracer='LRG', project='', version='holi-v1-altmtl', complete=False, imocks=[201], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', postprocess=['combine_regions'], **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    zranges = tools.propose_fiducial('zranges', tracer)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, imock=imocks[0]), imocks=imocks, combine_regions={'stats': ['mesh2_spectrum', 'mesh3_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'window_mesh3_spectrum'][2:4]}, mesh2_spectrum={'cut': True}, window_mesh2_spectrum={'cut': True})
    if complete:
        get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project, extra='complete')
    else:
        get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, project=project)
    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)



if __name__ == '__main__':

    mode = 'interactive'
    #mode = 'slurm'
    stats, postprocess = [], []
    stats = ['mesh2_spectrum', 'mesh3_spectrum']
    #stats = ['mesh3_spectrum']
    #stats = ['window_mesh2_spectrum']
    #stats = ['covariance_mesh2_spectrum']
    #stats = ['window_mesh3_spectrum']
    postprocess = ['combine_regions']
    #postprocess = ['rotation_mesh2_spectrum']
    imocks = np.arange(1001)
    imocks = [201]

    # stats_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    # version = 'holi-v1-altmtl'
    stats_dir  = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' / 
    # stats_dir = tools.base_stats_dir
    analysis = 'full_shape'
    project = f'{analysis}/base'
    version = 'holi-v3-altmtl'
    

    for tracer in ['LRG', 'ELG_LOPnotqso', 'QSO']:
        if False:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer, region='NGC', version=version), test_if_readable=False, imock=list(range(1001)))[:2]
            imocks = exists[1]['imock']
            rerun = []
            for zrange in tools.propose_fiducial('zranges', tracer, analysis=analysis):
                for kind in ['mesh2_spectrum', 'mesh3_spectrum']:
                    rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, kind=kind, stats_dir=stats_dir, project=project, tracer=tracer, region='GCcomb', weight='default-FKP', zrange=zrange, version=version), test_if_readable=True, imock=list(range(1001)))
                    rerun += [imock for imock in imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            imocks = sorted(set(rerun))

        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)
       
        run_stats_kws = dict(tracer=tracer, project=project, version=version, stats_dir=stats_dir, stats=stats, analysis=analysis, weight=weight, noric=noric)
        if any('window' in stat for stat in stats):
            _imocks = [201]
            nbatches = 1
            tasks = []
            for ibatch in range(nbatches):
                task = get_run_stats()(tracer, project=project, version=version, imocks=_imocks, stats_dir=stats_dir, stats=stats, ibatch=(ibatch, nbatches))
                tasks.append(task)
            if nbatches >= 1:
                # Add dependence on other tasks
                get_run_stats()(tracer, project=project, version=version, imocks=_imocks, stats_dir=stats_dir, stats=stats, ibatch=nbatches, tasks=tasks)
        elif any('covariance' in stat for stat in stats):
            get_run_stats()(tracer, project=project, version=version, imocks=[201], stats_dir=stats_dir, stats=stats)
        elif stats:
            batch_imocks = np.array_split(imocks, max(len(imocks) // 10, 1)) if len(imocks) else []
            for _imocks in batch_imocks:
                get_run_stats()(tracer, project=project, version=version, imocks=_imocks, stats_dir=stats_dir, stats=stats)
        if postprocess:
            postprocess_stats(tracer, project=project, version=version, imocks=imocks, stats_dir=stats_dir, postprocess=postprocess)
