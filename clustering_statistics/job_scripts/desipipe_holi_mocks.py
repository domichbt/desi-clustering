"""
Script to create and spawn desipipe tasks to compute clustering measurements on HOLI mocks.
To create and spawn the tasks on NERSC, use the following commands:
```bash
salloc -N 1 -C "gpu&hbm80g" -t 04:00:00 --gpus 4 --qos interactive --account desi_g
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
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))
tmw = tm.clone(scheduler=dict(max_workers=1), provider=dict(provider='nersc', time='00:10:00',
                mpiprocs_per_worker=2250, nodes_per_worker=25, output=output, error=error, stop_after=1, constraint='cpu'))

def run_stats(tracer='LRG', project='', version='holi-v1-altmtl', onthefly=None, imocks=[201], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], analysis='full_shape', ibatch=None, postprocess=None, zranges=None, **kwargs):
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
        regions = ['NGC', 'SGC']
        for region in regions:
            options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, imock=imock), mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True}, window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})

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
        postprocess_options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
        postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **postprocess_options)


def postprocess_stats(tracer='LRG', analysis='full_shape', project='', version='holi-v1-altmtl', onthefly=None, imocks=[201], stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], postprocess=['combine_regions'], zranges=None, **kwargs):
    from clustering_statistics import postprocess_stats_from_options
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer, analysis=analysis)
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, imock=imocks[0]), imocks=imocks, combine_regions={'stats': stats}, mesh2_spectrum={'cut': True, 'auw': True}, window_mesh2_spectrum={'cut': True})
    stats_dir_kws = dict(stats_dir=stats_dir, project=project)
    if onthefly == 'complete':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='complete', **stats_dir_kws)
    elif onthefly == 'reshuffle':
        get_stats_fn = functools.partial(tools.get_stats_fn, extra='reshuffle', **stats_dir_kws)
    else:
        get_stats_fn = functools.partial(tools.get_stats_fn, **stats_dir_kws)

    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)



if __name__ == '__main__':

    # # mode = 'interactive'
    # mode = 'slurm'
    # stats, postprocess = [], []
    # stats = ['mesh2_spectrum', 'mesh3_spectrum']
    #stats = ['mesh3_spectrum']
    #stats = ['window_mesh2_spectrum']
    #stats = ['covariance_mesh2_spectrum']
    #stats = ['window_mesh3_spectrum']
    # postprocess = ['combine_regions']
    #postprocess = ['rotation_mesh2_spectrum']
    # imocks = np.arange(1001)
    # imocks = [0]

    # stats_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    # version = 'holi-v1-altmtl'
    # stats_dir  = Path(os.getenv('SCRATCH')) / 'cai-dr2-benchmarks' 
    # to run job
    # mode = 'slurm'
    mode = 'interactive'
    stats, postprocess = [], []
    stats = ['mesh2_spectrum', 'mesh3_spectrum']
    postprocess = ['combine_regions']
    imocks = np.arange(1000)
    stats_dir = tools.base_stats_dir
    analysis = 'full_shape'
    project = f'{analysis}/base'
    version = 'holi-v3-altmtl'
    # weights = ['default-FKP','default-noimsys-FKP']
    weight = 'default-FKP'
    onthefly = None
    zranges = None

    for tracer in ['LRG', 'ELG_LOPnotqso', 'QSO'][1:]:
        if False:
            # Rerun stats for high-z
            if 'LRG' in tracer:
                zranges = [(0.8,1.1)]
            elif 'ELG' in tracer:
                zranges = [(1.1,1.6)]
            else:
                zranges = None

        if True:
            exists, missing = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_catalog_fn, tracer=tracer, region='NGC', version=version), test_if_readable=False, imock=list(range(1001)))[:2]
            imocks = exists[1]['imock']
            rerun = []
            for zrange in tools.propose_fiducial('zranges', tracer, analysis=analysis):
                for kind in ['mesh2_spectrum', 'mesh3_spectrum']:
                    stats_kws = dict(basis='sugiyama-diagonal', kind=kind, stats_dir=Path(str(stats_dir).replace('global','dvs_ro')), tracer=tracer, region='GCcomb', weight='default-FKP', zrange=zrange, version=version, project=project)
                    rexists, missing, unreadable = tools.checks_if_exists_and_readable(get_fn=functools.partial(tools.get_stats_fn, **stats_kws), test_if_readable=True, imock=list(range(1001)))
                    rerun += [imock for imock in imocks if (imock in unreadable[1]['imock']) or (imock not in rexists[1]['imock'])]
            imocks = sorted(set(rerun))
       
        def get_run_stats():
            _tm = tm80
            if tracer in ['LRG']:
                _tm = tm
            if any('window_mesh3' in stat for stat in stats):
                _tm = tmw
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        run_stats_kws = dict(tracer=tracer, stats_dir=stats_dir, project=project, version=version, stats=stats, analysis=analysis, onthefly=onthefly, zranges=zranges, postprocess=postprocess)
        if False:
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
                batch_imocks = np.array_split(imocks, max(len(imocks) // 10, 1)) if len(imocks) else []
                for _imocks in batch_imocks:
                    get_run_stats()(imocks=_imocks, **run_stats_kws)
        # if postprocess:
        #     postprocess_stats(imocks=imocks, **run_stats_kws)
