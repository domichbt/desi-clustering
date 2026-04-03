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

import os
import numpy as np
import functools
from pathlib import Path
# Import job queue management and task scheduling tools from desipipe (when running in batch mode)
from desipipe import Queue, Environment, TaskManager, spawn, setup_logging

# Import clustering statistics and logging utilities
from clustering_statistics import tools
setup_logging()

from mpi4py import MPI
mpicomm = MPI.COMM_WORLD

# Create a job queue named 'data_bao' for managing computation tasks
queue = Queue('data_bao')
# Clear any previous tasks in the queue (keep running jobs with kill=False)
queue.clear(kill=False)

# Define SLURM output/error log file paths for job monitoring
output, error = './slurm_outputs/data_bao/slurm-%j.out', './slurm_outputs/data_bao/slurm-%j.err'
kwargs = {}
# Set up NERSC HPC environment configuration for job execution
environ = Environment('nersc-cosmodesi')
# Create task manager to orchestrate distributed job execution
tm = TaskManager(queue=queue, environ=environ)
# Configure default task manager: 10 parallel workers, 4 MPI processes per worker
# GPU nodes with 1.5 hour time limit, stop after 1 failure
tm = tm.clone(scheduler=dict(max_workers=10), provider=dict(provider='nersc', time='01:30:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu'))
# Configure high-memory GPU task manager (80GB HBM): 2 hour time limit for larger jobs
tm80 = tm.clone(provider=dict(provider='nersc', time='02:00:00',
                            mpiprocs_per_worker=4, output=output, error=error, stop_after=1, constraint='gpu&hbm80g'))

def run_stats(version='data-dr2-v1.1', tracer='LRG', weight_type='weight-FKP', zranges=None, stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', stats=['mesh2_spectrum'], ibatch=None, **kwargs):
    # Everything inside this function will be executed on the compute nodes;
    # This function must be self-contained; and cannot rely on imports from the outer scope.
    import os
    import sys
    import functools
    from pathlib import Path
    # Import JAX for GPU-accelerated array operations
    import jax
    from jax import config
    # Enable 64-bit precision for accurate clustering calculations
    config.update('jax_enable_x64', True)
    # Allocate 90% of available GPU memory to JAX arrays
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    # Initialize JAX distributed computing across MPI processes
    try: jax.distributed.initialize()
    except RuntimeError: print('Distributed environment already initialized')
    else: print('Initializing distributed environment')
    # Import clustering statistics computation functions and logging
    from clustering_statistics import tools, setup_logging, compute_stats_from_options, fill_fiducial_options
    setup_logging()
    # Initialize cache dictionary to store intermediate results across regions
    cache = {}
    # If redshift ranges not provided, use fiducial values from tools
    if zranges is None:
        zranges = tools.propose_fiducial('zranges', tracer)
    # Create partial function with stats directory preset for cleaner calls
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    # Loop over Northern and Southern galactic caps
    for region in ['NGC', 'SGC']:
        # Configuration for particle pair correlation function (optional fine binning)
        # battrs: custom binning in separation s and line-of-sight angle mu
        battrs = None #dict(s=np.linspace(0., 150., 151), mu=(np.linspace(-1., 1., 201), 'midpoint'))
        # Particle pair correlation options: 60-point jackknife for covariance estimation
        particle2_correlation = {'battrs': battrs, 'jackknife': {'nsplits': 60}}
        # Build comprehensive options dictionary for all statistics
        # Catalog: data version, tracer type, redshift range, region, and weights
        # mesh2_spectrum: power spectrum multipoles (P(k) monopole, quadrupole, hexadecapole)
        # window_mesh2_spectrum: survey window function (mode coupling effects)
        # particle2_correlation: direct pair counting (optionally with jackknife error bars)
        # recon_particle2_correlation: same but on reconstruction-improved positions (BAO)
        # window_mesh3_spectrum: bispectrum window (batched computation with ibatch)
        options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, region=region, weight_type=weight_type), mesh2_spectrum={}, window_mesh2_spectrum={}, particle2_correlation=particle2_correlation, recon_particle2_correlation=particle2_correlation, window_mesh3_spectrum={'ibatch': ibatch} if isinstance(ibatch, tuple) else {'computed_batches': ibatch})
        # Fill in missing options with default/fiducial values from tools
        options = fill_fiducial_options(options)
        # Compute all requested statistics
        compute_stats_from_options(stats, get_stats_fn=get_stats_fn, cache=cache, **options)


def postprocess_stats(version='data-dr2-v1.1', tracer='LRG', weight='default-FKP', stats_dir=Path(os.getenv('SCRATCH')) / 'measurements', postprocess=['combine_regions'], **kwargs):
    # Post-processing step: combine measurements from NGC and SGC regions
    from clustering_statistics import postprocess_stats_from_options
    # Get fiducial redshift ranges for tracer
    zranges = tools.propose_fiducial('zranges', tracer)
    # Create partial function with stats directory preset
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir)
    # Particle pair correlation options: 60-point jackknife
    particle2_correlation = {'jackknife': {'nsplits': 60}}
    # Build post-processing options: combine statistics across regions
    # List statistics to combine: power spectrum, bispectrum, windows, covariance, pair correlations
    options = dict(catalog=dict(version=version, tracer=tracer, zrange=zranges, weight=weight), combine_regions={'stats': ['mesh2_spectrum', 'mesh3_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'window_mesh3_spectrum', 'recon_particle2_correlation']}, particle2_correlation=particle2_correlation, recon_particle2_correlation=particle2_correlation)
    # Execute post-processing: combines NGC+SGC, computes weighted averages, propagates covariance
    postprocess_stats_from_options(postprocess, get_stats_fn=get_stats_fn, **options)


if __name__ == '__main__':

    # Set execution mode: 'interactive' runs locally, anything else uses desipipe task queue
    mode = 'interactive'
    # List of statistics to compute; empty list [:0] disables computation
    # mesh2_spectrum: measure P(k) using FFT-based FKP method
    # window_mesh2_spectrum: compute survey window function via random catalogs
    # covariance_mesh2_spectrum: estimate power spectrum covariance matrix
    # recon_particle2_correlation: pair counting on BAO-reconstructed positions
    stats = ['mesh2_spectrum', 'window_mesh2_spectrum', 'covariance_mesh2_spectrum', 'recon_particle2_correlation'][:0]
    # combine_regions: merge NGC and SGC measurements into GCcomb estimates
    postprocess = ['combine_regions'][:1]

    # Output directory for measurement results (SCRATCH filesystem for performance)
    stats_dir = Path(f'/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    # Catalog version
    #version = 'data-dr2-v1.1'
    version = 'data-dr1-v1.5'

    # Loop over tracer types: BGS (Bright Galaxy Survey), LRG (Luminous Red Galaxy), ELG (Emission Line Galaxy), QSO (Quasar)
    # [1:2] selects only LRG; change to [:] to process all tracers
    for tracer in ['BGS', 'LRG', 'ELG', 'QSO'][1:2]:
        # Get full tracer name including version suffix (e.g., 'LRG_0' for redshift bin 0)
        tracer = tools.get_full_tracer(tracer, version=version)
        # Get fiducial redshift ranges for this tracer; [:1] takes only first bin
        zranges = tools.propose_fiducial('zranges', tracer)[:1]

        def get_run_stats():
            # Dynamically select task manager based on computation type and tracer
            _tm = tm80
            # LRG uses standard task manager (fits on GPU with 40GB memory)
            if tracer in ['LRG']:
                _tm = tm
            # Return function reference (desipipe app wrapper if batch mode, local if interactive)
            return run_stats if mode == 'interactive' else _tm.python_app(run_stats)

        # Execute statistics computation if requested (stats list not empty)
        if stats:
            get_run_stats()(version=version, tracer=tracer, zranges=zranges, stats_dir=stats_dir, stats=stats)
        # Execute post-processing if requested (combine regions, etc.)
        if postprocess:
            postprocess_stats(version=version, tracer=tracer, zranges=zranges, stats_dir=stats_dir, postprocess=postprocess)