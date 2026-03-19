"""
salloc -N 1 -C "gpu&hbm80g" -t 02:00:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh new
srun -n 4 python tests/test-recon.py
"""
import os
import sys
import logging
import functools
from pathlib import Path

import numpy as np
import lsstypes as types
import mpytools as mpy

from clustering_statistics import tools, setup_logging, compute_stats_from_options
from clustering_statistics.tools import fill_fiducial_options, _merge_options, Catalog, setup_logging, compute_fkp_effective_redshift

logger = logging.getLogger('test_recon')

def _make_list_zrange(zranges):
    if np.ndim(zranges[0]) == 0:
        zranges = [zranges]
    return list(zranges)

def recon_output(get_catalog_fn=None, get_stats_fn=tools.get_stats_fn,
                                read_clustering_catalog=tools.read_clustering_catalog,
                                read_full_catalog=tools.read_full_catalog, analysis='full_shape', cache=None, **kwargs):
    cache = cache or {}
    options = fill_fiducial_options(kwargs, analysis=analysis)
    catalog_options = options['catalog']
    tracers = list(catalog_options.keys())

    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}

    if get_catalog_fn is not None:
        read_clustering_catalog = functools.partial(read_clustering_catalog, get_catalog_fn=get_catalog_fn)
        read_full_catalog = functools.partial(read_full_catalog, get_catalog_fn=get_catalog_fn)

    with_catalogs = True
    data, randoms = {}, {}
    with_stats_blinding = False

    if with_catalogs:
        for tracer in tracers:
            _catalog_options = dict(catalog_options[tracer])
            _catalog_options['zrange'] = (min(zrange[0] for zrange in zranges[tracer]), max(zrange[1] for zrange in zranges[tracer]))
            if any(name in _catalog_options.get('weight', '') for name in ['bitwise', 'compntile']):
                # sets NTILE-MISSING-POWER (missing_power) and per-tile completeness (completeness)
                _catalog_options['binned_weight'] = read_full_catalog(kind='parent_data', **_catalog_options, attrs_only=True)


            recon_options = options['recon'][tracer]
            # pop as we don't need it anymore
            _catalog_options |= {key: recon_options.pop(key) for key in list(recon_options) if key in ['nran', 'zrange']}

            with_stats_blinding |= tools.check_if_stats_requires_blinding(analysis=analysis, **_catalog_options)
            if isinstance(_catalog_options.get('complete', None), dict):
                _catalog_options.setdefault('reshuffle', {})  # to pass on complete data
            data[tracer] = read_clustering_catalog(kind='data', **_catalog_options, concatenate=True)
            #_catalog_options.pop('complete', None)
            #_catalog_options.pop('reshuffle', None)
            randoms[tracer] = read_clustering_catalog(kind='randoms', **_catalog_options, cache=cache, concatenate=False)

    import jax
    from jaxpower import create_sharding_mesh
    from clustering_statistics.recon_tools import compute_reconstruction
    with create_sharding_mesh() as sharding_mesh:

        # data_rec, randoms_rec = {}, {}
        for tracer in tracers:
            recon_options = dict(options['recon'][tracer])
            # local sizes to select positions
            data[tracer]['POSITION_REC'], randoms_rec_positions = compute_reconstruction(lambda: {'data': data[tracer], 'randoms': Catalog.concatenate(randoms[tracer])}, **recon_options)
            start = 0
            for random in randoms[tracer]:
                size = len(random['POSITION'])
                random['POSITION_REC'] = randoms_rec_positions[start:start + size]
                start += size
            randoms[tracer] = randoms[tracer][:catalog_options[tracer]['nran']]  # keep only relevant random files

        for zvals in zip(*(zranges[tracer] for tracer in tracers)):
            zrange = dict(zip(tracers, zvals))

            def get_zcatalog(catalog, zrange):
                mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])
                return catalog[mask]

            zdata, zrandoms = {}, {}
            for tracer in tracers:
                zdata[tracer] = get_zcatalog(data[tracer], zrange[tracer])
                zrandoms[tracer] = [get_zcatalog(random, zrange[tracer]) for random in randoms[tracer]]
            # fn_catalog_options = {tracer: catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}
    return zdata, zrandoms

def compute_reconstruction_cpu(get_data_randoms, mattrs=None, mode='recsym', bias=2.0, smoothing_radius=15., mpicomm=None):
    """
    Run density field reconstruction using :mod:`pyrecon` (CPU/MPI).

    Parameters
    ----------
    get_data_randoms : callable
        Function that returns dict of 'data' and 'randoms' catalogs (randoms concatenated).
        See :func:`compute_reconstruction` for details.
    mattrs : dict, optional
        Mesh attributes: 'cellsize', 'boxsize', 'boxpad'. If None, defaults are used.
    mode : {'recsym', 'reciso'}, optional
        Reconstruction mode. 'recsym' removes large-scale RSD from randoms; 'reciso' does not.
    bias : float, optional
        Linear tracer bias.
    smoothing_radius : float, optional
        Smoothing radius in Mpc/h for the density field.
    mpicomm : MPI communicator, optional
        MPI communicator. Defaults to mpy.CurrentMPIComm.get().
    """
    from pyrecon import IterativeFFTReconstruction
    from cosmoprimo.fiducial import DESI

    mattrs = mattrs or {}
    cellsize = mattrs.get('cellsize', smoothing_radius / 3.)
    boxsize = mattrs.get('boxsize', None)
    boxpad = mattrs.get('boxpad', None)

    if mpicomm is None:
        mpicomm = mpy.CurrentMPIComm.get()

    catalogs = get_data_randoms()
    data = catalogs['data']
    randoms = catalogs['randoms']

    cosmo = DESI()
    data_positions = data['POSITION']
    data_weights = data['INDWEIGHT']

    z = data['Z']
    zeff = float(mpicomm.allreduce(np.sum(z * data_weights)) / mpicomm.allreduce(np.sum(data_weights)))
    f = cosmo.growth_rate(zeff)
    if mpicomm.rank == 0:
        logger.info('f = %.3f, bias = %.3f at zeff = %.3f with smoothing_radius = %.3f, cellsize = %.3f', f, bias, zeff, smoothing_radius, cellsize)

    recon_kwargs = dict(f=f, bias=bias, positions=data_positions, cellsize=cellsize, position_type='pos', mpicomm=mpicomm)
    if boxsize is not None:
        recon_kwargs['boxsize'] = boxsize
    elif boxpad is not None:
        recon_kwargs['boxpad'] = boxpad

    recon = IterativeFFTReconstruction(**recon_kwargs)
    recon.assign_data(data_positions, data_weights)
    recon.assign_randoms(randoms['POSITION'], randoms['INDWEIGHT'])
    recon.set_density_contrast(smoothing_radius=smoothing_radius)
    recon.run()

    if mpicomm.rank == 0:
        logger.info('Reconstruction done, reading shifted positions.')

    data_positions_rec = recon.read_shifted_positions(data_positions)
    assert mode in ['recsym', 'reciso']
    if mode == 'recsym':
        randoms_positions_rec = recon.read_shifted_positions(randoms['POSITION'])
    else:
        randoms_positions_rec = recon.read_shifted_positions(randoms['POSITION'], field='disp')

    return data_positions_rec, randoms_positions_rec


def _make_list_zrange(zranges):
    if np.ndim(zranges[0]) == 0:
        zranges = [zranges]
    return list(zranges)

def recon_output_cpu(get_catalog_fn=None, get_stats_fn=tools.get_stats_fn,
                                read_clustering_catalog=tools.read_clustering_catalog,
                                read_full_catalog=tools.read_full_catalog, analysis='full_shape', **kwargs):
    kwargs = fill_fiducial_options(kwargs, analysis=analysis)
    catalog_options = kwargs['catalog']
    tracers = list(catalog_options.keys())

    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}

    if get_catalog_fn is not None:
        read_clustering_catalog = functools.partial(read_clustering_catalog, get_catalog_fn=get_catalog_fn)
        read_full_catalog = functools.partial(read_full_catalog, get_catalog_fn=get_catalog_fn)

    data, randoms = {}, {}
    for tracer in tracers:
        _catalog_options = dict(catalog_options[tracer])
        _catalog_options['zrange'] = (min(zrange[0] for zrange in zranges[tracer]), max(zrange[1] for zrange in zranges[tracer]))
        recon_options = kwargs['recon'][tracer]
        _catalog_options |= {key: recon_options.pop(key) for key in list(recon_options) if key in ['nran', 'zrange']}
        if any(name in catalog_options.get('weight', '') for name in ['bitwise', 'compntile']):
            # sets NTILE-MISSING-POWER (missing_power) and per-tile completeness (completeness)
            _catalog_options['binned_weight'] = read_full_catalog(kind='parent_data', **_catalog_options, attrs_only=True)
        logger.info("Catalog options for tracer %s: %s", tracer, _catalog_options)

        data[tracer] = read_clustering_catalog(kind='data', **_catalog_options, concatenate=True)
        randoms[tracer] = read_clustering_catalog(kind='randoms', **_catalog_options, concatenate=False)

    mpicomm = mpy.CurrentMPIComm.get()

    for tracer in tracers:
        recon_options = dict(kwargs['recon'][tracer])
        logger.info("Recon options for tracer %s: %s", tracer, recon_options)

        data_positions_rec, randoms_rec_positions = compute_reconstruction_cpu(lambda: {'data': data[tracer], 'randoms': Catalog.concatenate(randoms[tracer])}, mpicomm=mpicomm, **recon_options)
        data[tracer]['POSITION_REC'] = data_positions_rec
        start = 0
        for random in randoms[tracer]:
            size = len(random['POSITION'])
            random['POSITION_REC'] = randoms_rec_positions[start:start + size]
            start += size
        randoms[tracer] = randoms[tracer][:catalog_options[tracer]['nran']]  # keep only relevant random files

    for zvals in zip(*(zranges[tracer] for tracer in tracers)):
        zrange = dict(zip(tracers, zvals))

        def get_zcatalog(catalog, zrange):
            mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])
            return catalog[mask]

        zdata, zrandoms = {}, {}
        for tracer in tracers:
            zdata[tracer] = get_zcatalog(data[tracer], zrange[tracer])
            zrandoms[tracer] = [get_zcatalog(random, zrange[tracer]) for random in randoms[tracer]]
    return zdata, zrandoms

def test_recon_output():
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=451, nran=2)
            catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
            # recon_output0(stat, catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, particle2_correlation={})
            
            zdata, zrandoms = recon_output(catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, particle2_correlation={}, recon={'bias':2.0})

    # show some structure of data and randoms
        print(type(zdata[tracer]), type(zrandoms[tracer]), len(zrandoms[tracer]), len(zrandoms[tracer][0]), type(zrandoms[tracer][0][0]))
        print(zdata[tracer][0].keys(), zdata[tracer][0]['POSITION_REC'].shape)
        print(zrandoms[tracer][0][0].keys(), zrandoms[tracer][0][0]['POSITION_REC'].shape)

        zrange_text = f"z{zrange[0]}-{zrange[1]}"
        zdata[tracer].write(stats_dir / f'jax_recon_{tracer}_{zrange_text}_{region}_clustering.dat.h5')
        for i, zrandom in enumerate(zrandoms[tracer]):
            zrandom.write(stats_dir / f'jax_recon_{tracer}_{i}_{zrange_text}_{region}_clustering.ran.h5')
    
    return zdata, zrandoms

def test_recon_clustering(stat=['recon_particle2_correlation', 'recon_mesh2_spectrum'],
                          to_test=['boxsize', 'nran', 'cellsize']):
    checks = {
        'cellsize': check_cellsize_recon,
        'boxsize':  check_boxsize_recon,
        'nran':     check_nran_recon,
        'sigma':    get_sigma_recon,
    }
    for name in to_test:
        if name in checks:
            checks[name](stat=stat)

def check_cellsize_recon(stat=['recon_particle2_correlation', 'recon_mesh2_spectrum']):
    """Run recon measurements with varying reconstruction cell size to check stability."""
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    cellsizes = [4,6]#[8, 10, 12]
    nran = 18
    boxsize = 10000.
    for tracer in ['ELG_LOPnotqso']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC']:
            for cellsize in cellsizes:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=451, nran=nran)
                catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
                extra = f'nran{nran:d}_cellsize{cellsize:.2f}_boxsize{boxsize:.0f}'
                compute_stats_from_options(stat, catalog=catalog_options, mattrs={'cellsize': cellsize, 'boxsize': boxsize},
                                           get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir, extra=extra),
                                           recon_mesh2_spectrum={}, recon_particle2_correlation={})


def check_boxsize_recon(stat=['recon_particle2_correlation', 'recon_mesh2_spectrum']):
    """Run recon measurements with varying box size to check stability."""
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    boxsizes = {'ELG_LOPnotqso': [6000., 7000., 8000., 9000., 10000.]}
    cellsize = 7.8  
    nran = 18
    for tracer in boxsizes:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC']:
            for boxsize in boxsizes[tracer]:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=451, nran=nran)
                catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
                extra = f'boxsize{boxsize:.0f}'
                compute_stats_from_options(stat, catalog=catalog_options, mattrs={'boxsize': boxsize, 'cellsize': cellsize},
                                           get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir, extra=extra),
                                           recon_mesh2_spectrum={}, recon_particle2_correlation={})
                
def get_sigma_recon(stat=['recon_particle2_correlation', 'recon_mesh2_spectrum']):
    """Run recon measurements with varying box size to check stability."""
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    boxsizes = {'ELG_LOPnotqso':[10000.]}#'LRG': [10000.], 
    cellsize = 7.8  
    nran = 18
    imocks = range(451, 457)
    for tracer in boxsizes:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        boxsize = boxsizes[tracer][0]
        for region in ['NGC','SGC']:# 
            for imock in imocks:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=imock, nran=nran)
                catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
                extra = f'nran{nran:d}_cellsize{cellsize:.2f}_boxsize{boxsize:.0f}'
                compute_stats_from_options(stat, catalog=catalog_options, mattrs={'boxsize': boxsize, 'cellsize': cellsize},
                                           get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir, extra=extra),
                                           recon_mesh2_spectrum={}, recon_particle2_correlation={})


def check_nran_recon(stat=['recon_particle2_correlation', 'recon_mesh2_spectrum']):
    """Run recon measurements with varying number of randoms to check stability."""
    stats_dir = Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks'
    nrans = [2, 4, 6, 8, 10, 12, 14, 16]#
    cellsize = 7.8
    boxsize = 10000.
    for tracer in ['ELG_LOPnotqso']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC']:
            for nran in nrans:
                catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=451, nran=nran)
                catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
                extra = f'nran{nran:d}_cellsize{cellsize:.2f}_boxsize{boxsize:.0f}'
                compute_stats_from_options(stat, catalog=catalog_options, mattrs={'boxsize': boxsize, 'cellsize': cellsize},
                                           get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir, extra=extra),
                                           recon_mesh2_spectrum={}, recon_particle2_correlation={})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=False, help='Run on CPU (pyrecon/MPI), skipping JAX imports and initialization')
    args = parser.parse_args()

    setup_logging()
    if args.cpu:
        recon_output_cpu(catalog=dict(version='holi-v1-altmtl', tracer='LRG', region='NGC', imock=451, nran=2), recon={'bias': 2.0})
    else:
        import jax
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
        jax.distributed.initialize()
        # data, randoms = test_recon_output()
        test_recon_clustering(stat=['recon_particle2_correlation', 'recon_mesh2_spectrum'], to_test=['cellsize'])#'boxsize', sigma']


