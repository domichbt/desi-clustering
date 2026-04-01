"""
Script to create and spawn desipipe tasks to compute clustering measurements on glam-uchuu mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
export MPICH_MPIIO_DVS_STRIPE_WIDTH=1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh new
export PYTHONPATH=$HOME/cai-dr2-clustering-products/:$PYTHONPATH
python desipipe_glam-uchuu_mocks.py         # create the list of tasks
desipipe tasks  -q glam-uchuu_mocks         # check the list of tasks
desipipe spawn  -q glam-uchuu_mocks --spawn # spawn the jobs
desipipe queues -q glam-uchuu_mocks         # check the queue
```
"""
import os
from pathlib import Path
import functools

import numpy as np
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

from clustering_statistics import tools

setup_logging()

queue = Queue('glam-uchuu_mocks')
queue.clear(kill=False)

output, error = 'slurm_outputs/glam-uchuu_mocks/slurm-%j.out', 'slurm_outputs/glam-uchuu_mocks/slurm-%j.err'
kwargs = {}
environ = Environment(command='source /global/common/software/desi/users/adematti/cosmodesi_environment.sh new')
tm = TaskManager(queue=queue, environ=environ)
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))

def run_stats(tracer='LRG', project='', version='glam-uchuu-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', analysis='full_shape', regions=['NGC','SGC'], ibatch=None, postprocess=None, zranges=None, **kwargs):
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
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options, postprocess_stats_from_options
    setup_logging()

    cache = {}
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    for imock in imocks:
        for region in regions:
            mesh2_spectrum = {'cut': True, 
                              'auw': True if 'altmtl' in version and onthefly is None else None,
                              'optimal_weights': functools.partial(tools.compute_fiducial_png_weights, tracer=tracer) if 'oqe' in weight else None}
            window_mesh2_spectrum = {'cut': True, 
                                     'optimal_weights': functools.partial(tools.compute_fiducial_png_weights, tracer=tracer) if 'oqe' in weight else None}
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight=weight, imock=imock), 
                           mesh2_spectrum=mesh2_spectrum, window_mesh2_spectrum=window_mesh2_spectrum, 
                           window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})

            stats_dir_kws = dict(stats_dir=stats_dir, project=project)
            if onthefly == 'complete':
                options['catalog']['complete'] = {}
                get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
            elif onthefly == 'reshuffle':
                options['catalog']['reshuffle'] = {'merged_data_fn': tools.get_catalog_fn(kind='data', **(options['catalog'] | dict(region='ALL')))}
                get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
            else:
                get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)

            options = fill_fiducial_options(options)
            if True: #onthefly:
                for tracer in options['catalog']:
                    options['catalog'][tracer]['expand'] = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=options['catalog'][tracer]['nran']), 'from_data': ['Z', 'WEIGHT_SYS', 'FRAC_TLOBS_TILES']}
            compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, **options)

    # postprocess
    if postprocess:
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
        postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **postprocess_options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='glam-uchuu-v2-altmtl', onthefly=None, imocks=[150], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], weight='default-FKP', postprocess=['combine_regions'], zranges=None, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
    stats_dir_kws = dict(stats_dir=stats_dir, project=project)
    if onthefly == 'complete':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
    elif onthefly == 'reshuffle':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
    else:
        get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)

    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)



if __name__ == '__main__':

    stats, postprocess = [], []
    version  = 'glam-uchuu-v2-altmtl'
    check_for_existing_measurements = True
    
    # run on interactive node
    # mode = 'interactive'
    # imocks2run = 150 + np.arange(1)
    # stats_dir  = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' 
    
    # to run job
    mode = 'slurm'
    imocks2run = 150 + np.arange(50)
    stats_dir  = tools.base_stats_dir

    # run fiducial full_shape
    # stats       = ['mesh2_spectrum', 'mesh3_spectrum']
    # postprocess = ['combine_regions']
    # analysis = 'full_shape'
    # project  = f'{analysis}/base'
    # weight   = 'default-FKP'
    # regions  = ['NGC','SGC']
    # max_mocks_per_batch = 10

    # run data_splits for lensing group with full_shape setup 
    stats   = ['mesh2_spectrum']
    analysis = 'full_shape'
    project = f'{analysis}/data_splits'
    weight  = 'default-FKP'
    regions = ['N','NGCnoN','S','SGCnoDES','SnoDES','DES','ACT_DR6','PLANCK_PR4'] # ,'GAL040','GAL060']
    # regions = ['GAL040','GAL060']
    max_mocks_per_batch = 5 

    # run fiducial local_png
    # stats       = ['mesh2_spectrum']
    # postprocess = ['combine_regions']
    # analysis = 'local_png'
    # project  = f'{analysis}/base'
    # weight   = 'default-FKP-oqe'
    # regions  = ['NGC','SGC']
    # max_mocks_per_batch = 10

    onthefly = None
    zranges  = None
    
    for tracer in ['LRG', 'ELG_LOPnotqso', 'QSO']:
        if check_for_existing_measurements:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer, region='NGC', version=version), test_if_readable=False, imock=imocks2run)[:2]
            imocks = exists[1]['imock']
            rerun = []
            for zrange in tools.propose_fiducial('zranges', tracer, analysis=analysis):
                for kind in stats:
                    stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')), 
                                     tracer=tracer, region=regions[-1], weight=weight, zrange=zrange, version=version, project=project)
                    rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, **stats_kws), test_if_readable=True, imock=imocks2run)
                    rerun += [imock for imock in imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            imocks = sorted(set(rerun))
        else:
            imocks = imocks2run
       
        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, onthefly=onthefly, zranges=zranges, regions=regions, weight=weight, postprocess=postprocess)
        if True:
            if any('window' in stat for stat in stats):
                _imocks = [201]
                nbatches = 1
                tasks = []
                for ibatch in range(nbatches):
                    task = get_run_stats()(imocks=_imocks, ibatch=(ibatch, nbatches), **run_stats_kws)
                    tasks.append(task)
                if nbatches >= 1:
                    # Add dependence on other tasks
                    get_run_stats()(imocks=_imocks, ibatch=nbatches, tasks=tasks, **run_stats_kws)
            elif any('covariance' in stat for stat in stats):
                get_run_stats()(imocks=[201], **run_stats_kws)
            elif stats:
                batch_imocks = np.array_split(imocks, max(len(imocks) // max_mocks_per_batch, 1)) if len(imocks) else []
                for _imocks in batch_imocks:
                    get_run_stats()(imocks=_imocks, **run_stats_kws)
        # if postprocess:
        #     postprocess_stats(imocks=imocks, **run_stats_kws)
