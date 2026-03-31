"""
Configuration-space 2-point clustering measurements.

Main functions
--------------
* `compute_particle2_correlation`: Measure the cutsky 2PCF from pair counts (includes jackknife utility).
* `compute_angular_upweights`: Derive angular upweights for fiber-collision mitigation.
* `compute_box_particle2_correlation`: Measure the 2PCF in cubic boxes.
"""

import logging
from functools import partial

import numpy as np
import jax

import lsstypes as types
from .tools import _format_bitweights


logger = logging.getLogger('correlation2')


def compute_angular_upweights(*get_data):
    """
    Compute angular upweights (AUW) from fibered and parent data catalogs.

    Parameters
    ----------
    get_data : callables
        Functions that return dict of 'fibered_data', 'parent_data' catalogs. Each catalog must contain 'RA', 'DEC', 'INDWEIGHT', and optionally 'BITWEIGHT'.

    Returns
    -------
    auw : ObservableTree
        Angular upweights as an ObservableTree with 'DD' leaf.
    """
    from cucount.jax import Particles, BinAttrs, WeightAttrs, setup_logging
    from cucount.jax import create_sharding_mesh
    from cucount.types import count2
    from lsstypes import ObservableLeaf, ObservableTree

    with create_sharding_mesh():
        all_fibered_data, all_parent_data = [], []

        def get_rdw(catalog):
            positions = (catalog['RA'], catalog['DEC'])
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        for _get_data in get_data:
            _data = _get_data()
            fibered_data = Particles(*get_rdw(_data['fibered_data']), positions_type='rd', exchange=True)
            parent_data = Particles(*get_rdw(_data['parent_data']), positions_type='rd', exchange=True)
            all_fibered_data.append(fibered_data)
            all_parent_data.append(parent_data)

        theta = 10**np.arange(-5, -1 + 0.1, 0.1)  # TODO: update
        battrs = BinAttrs(theta=theta)
        bitwise = None
        if all_fibered_data[0].get('bitwise_weight'):
            bitwise = dict(weights=all_fibered_data[0].get('bitwise_weight'))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')
        wattrs = WeightAttrs(bitwise=bitwise)
        DDfibered = count2(*all_fibered_data, battrs=battrs, wattrs=wattrs)['weight'].value()
        wattrs = WeightAttrs()
        DDparent = count2(*all_parent_data, battrs=battrs, wattrs=wattrs)['weight'].value()

    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    auw['DD'] = ObservableLeaf(value=np.where(DDfibered == 0., 1., DDparent / DDfibered), **kw)
    #auw['DR'] = ObservableLeaf(value=np.where(DRfibered == 0., 1., DRparent / DRfibered), **kw)
    auw = ObservableTree(list(auw.values()), pairs=list(auw.keys()))
    return auw


def compute_particle2_correlation(*get_data_randoms, auw=None, cut=None, battrs: dict=None, zeff: dict=None, jackknife: dict=None):
    """
    Compute two-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
        Each catalog must contain 'POSITION', 'INDWEIGHT', and optionally 'BITWEIGHT' for bitwise weights.
        Randoms and shifted catalogs can be lists of catalogs (for multiple randoms/shifted).
    auw : ObservableTree, optional
        Angular upweights to apply. If None, no angular upweights are applied.
    cut : bool, optional
        If provided, apply a theta-cut of (0, 0.05) in degress.
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs. If None, default bins are used. See cucount.jax.BinAttrs.
    zeff : dict, optional
        Optional arguments for computing effective redshift.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)

    Returns
    -------
    correlation : Count2Correlation
        Two-point correlation function as a Count2Correlation object.
    """
    import cucount
    from cucount.jax import Particles, BinAttrs, WeightAttrs, SelectionAttrs, MeshAttrs, setup_logging
    from cucount.types import count2
    from lsstypes import Count2, Count2Correlation, Count2JackknifeCorrelation

    if zeff is None: zeff = {'boxpad': 1.1, 'cellsize': 10.}
    kw_zeff = dict(zeff)
    if jackknife is None: jackknife = {}
    kw_jackknife = dict(jackknife)
    if kw_jackknife: kw_jackknife = {'mode': 'angular', 'nsplits': 60, 'nside': 512, 'random_state': 42} | kw_jackknife

    # First: effective redshift
    from .spectrum2_tools import prepare_jaxpower_particles, compute_fkp_effective_redshift
    from jaxpower import create_sharding_mesh

    def merge_randoms(catalog):
        if not isinstance(catalog, (tuple, list)):
            return catalog
        return catalog[0].concatenate(catalog)

    get_randoms = [lambda: {'randoms': merge_randoms(_get_data_randoms()['randoms'])} for _get_data_randoms in get_data_randoms]
    with create_sharding_mesh(meshsize=kw_zeff.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_randoms, mattrs=kw_zeff, add_randoms=['IDS'])
        all_randoms = [particles['randoms'] for particles in all_particles]
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, split=seed, resampler='cic', return_fraction=True)
        del all_particles, all_randoms

    from cucount.jax import create_sharding_mesh
    with create_sharding_mesh() as sharding_mesh:

        all_data, all_randoms, all_shifted = [], [], []

        def get_pw(catalog):
            positions = catalog['POSITION']
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        def _is_list(catalog):
            return isinstance(catalog, (tuple, list))

        def get_all_particles(catalog, subsampler=None, as_list=False):
            if as_list and not _is_list(catalog):
                catalog = [catalog]
            if _is_list(catalog):
                return [get_all_particles(catalog, subsampler=subsampler) for catalog in catalog]  # list of randoms
            positions, weights = get_pw(catalog)
            splits = None
            if subsampler is not None:
                splits = subsampler.label(positions).astype('i8')
            return Particles(positions, weights=weights, splits=splits, exchange=True)

        jackknife_particles = []
        for _get_data_randoms in get_data_randoms:
            data = cucount.numpy.Particles(*get_pw(_get_data_randoms()['data'].gather(mpiroot=None)))
            jackknife_particles.append(data)

        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))

        battrs = BinAttrs(**battrs)
        sattrs = None
        if cut is not None:
            sattrs = SelectionAttrs(theta=(0., 0.05))
            if jax.process_index() == 0:
                logger.info(f'Applying theta-cut {sattrs}.')
        bitwise = angular = None
        if data.get('bitwise_weight'):
            bitwise = dict(weights=data.get('bitwise_weight'))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')
        if auw is not None:
            angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
            if jax.process_index() == 0:
                logger.info(f'Applying AUW {angular}.')
        wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
        spattrs = None
        mattrs = None  # automatic setting for mesh

        subsampler = None
        if kw_jackknife:
            from cucount.jax import SplitAttrs
            from cucount.utils import KMeansSubsampler
            jackknife_particles = cucount.numpy.Particles.concatenate(jackknife_particles)
            subsampler = KMeansSubsampler(jackknife_particles, wattrs=wattrs, **kw_jackknife)
            spattrs = SplitAttrs(mode='jackknife', nsplits=subsampler.nsplits)
            #labels = subsampler.label(jackknife_particles)
            #print(np.bincount(labels))
            #exit()

        for _get_data_randoms in get_data_randoms:
            # data, randoms (optionally shifted) are tuples (positions, weights)
            _catalogs = _get_data_randoms()
            data = get_all_particles(_catalogs['data'], subsampler=subsampler)
            randoms = get_all_particles(_catalogs['randoms'], subsampler=subsampler, as_list=True)
            if _catalogs.get('shifted', None) is not None:
                shifted = get_all_particles(_catalogs['shifted'], subsampler=subsampler, as_list=True)
            else:
                shifted = [None] * len(randoms)
            all_data.append(data)
            all_randoms.append(randoms)
            all_shifted.append(shifted)

        if jax.process_index() == 0:
            logger.info(f'All particles on the device')

        _count2 = partial(count2, battrs=battrs, mattrs=mattrs, sattrs=sattrs, spattrs=spattrs)
        DD = _count2(*all_data, wattrs=wattrs)['weight']
        for i in range(len(all_data)):
            all_data[i] = all_data[i].clone(weights=wattrs(all_data[i]))   # clone data, with IIP weights (in case we provided bitwise weights)
        DS, SD, SS, RR = [], [], [], []
        iran = 0
        for all_randoms_i, all_shifted_i in zip(zip(*all_randoms, strict=True), zip(*all_shifted, strict=True), strict=True):
            if jax.process_index() == 0:
                logger.info(f'Processing random {iran:d}.')
            iran += 1
            RR.append(_count2(*all_randoms_i)['weight'])
            if all(shifted is not None for shifted in all_shifted_i):
                SS.append(_count2(*all_shifted_i)['weight'])
            else:
                all_shifted_i = all_randoms_i
                SS.append(RR[-1])
            DS.append(_count2(all_data[0], all_shifted_i[-1])['weight'])
            SD.append(_count2(all_shifted_i[0], all_data[-1])['weight'])

    DS, SD, SS, RR = (types.sum(XX) for XX in [DS, SD, SS, RR])
    correlation = (Count2JackknifeCorrelation if kw_jackknife else Count2Correlation)(estimator='landyszalay', DD=DD, DS=DS, SD=SD, SS=SS, RR=RR)
    correlation.attrs.update(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)
    return correlation


def compute_box_particle2_correlation(*get_data, battrs: dict=None, mattrs: dict=None, los='z'):
    """
    Compute two-point correlation function using :mod:`cucount.jax`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' (optionally 'shifted') catalogs.
        Each catalog must contain 'POSITION', 'INDWEIGHT'.
        Shifted catalogs can be lists of catalogs (for multiple randoms/shifted).
    battrs : dict, optional
        Bin attributes for cucount.jax.BinAttrs. If None, default bins are used. See cucount.jax.BinAttrs.
    mattrs : dict, array, optional
        Mesh attributes; typically a dictionary with 'boxsize' and 'boxcenter'.
    los : {'x', 'y', 'z', array-like}, optional
        Line-of-sight direction. If 'x', 'y', 'z' use fixed axes, or provide a 3-vector.

    Returns
    -------
    correlation : Count2Correlation
        Two-point correlation function as a Count2Correlation object.
    """
    from cucount.jax import Particles, BinAttrs, WeightAttrs, SelectionAttrs, MeshAttrs, setup_logging
    from cucount.types import count2, count2_analytic
    from lsstypes import Count2, Count2Correlation

    with jax.make_mesh((jax.device_count(),), axis_names=('x',), axis_types=(jax.sharding.AxisType.Auto,)):
        all_data, all_shifted = [], []

        def get_pw(catalog):
            positions = catalog['POSITION']
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        def _is_list(catalog):
            return isinstance(catalog, (tuple, list))

        def get_all_particles(catalog, as_list=False):
            if as_list and not _is_list(catalog):
                catalog = [catalog]
            if _is_list(catalog):
                return [get_all_particles(catalog) for catalog in catalog]  # list of randoms
            positions, weights = get_pw(catalog)
            return Particles(positions, weights=weights, exchange=True)

        for _get_data in get_data:
            # data (optionally shifted) are tuples (positions, weights)
            _catalogs = _get_data()
            data = get_all_particles(_catalogs['data'])  # data is not a list of catalogs
            if _catalogs.get('shifted', None) is not None:
                shifted = get_all_particles(_catalogs['shifted'], as_list=True)
            else:
                shifted = [None]
            all_data.append(data)
            all_shifted.append(shifted)
        if jax.process_index() == 0:
            logger.info(f'All particles on the device')

        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))

        battrs = BinAttrs(**battrs)
        wattrs = WeightAttrs()
        mattrs = mattrs or {}
        mattrs = MeshAttrs(*all_data, battrs=battrs, **mattrs)

        _count2 = partial(count2, battrs=battrs, mattrs=mattrs)
        DD = _count2(*all_data, wattrs=wattrs)['weight']
        for i in range(len(all_data)):
            all_data[i] = all_data[i].clone(weights=wattrs(all_data[i]))   # clone data, with IIP weights (in case we provided bitwise weights)
        RR = count2_analytic(battrs=battrs, mattrs=mattrs)

        DS, SD, SS = [], [], []
        iran = 0
        for all_shifted in zip(*all_shifted, strict=True):
            if jax.process_index() == 0:
                logger.info(f'Processing random {iran:d}.')
            if all(shifted is not None for shifted in all_shifted):
                SS.append(_count2(*all_shifted)['weight'])
                DS.append(_count2(all_data[0], all_shifted[-1])['weight'])
                SD.append(_count2(all_shifted[0], all_data[-1])['weight'])
                iran += 1

    if iran:
        DS, SD, SS = (types.sum(XX) for XX in [DS, SD, SS])
        correlation = Count2Correlation(estimator='landyszalay', DD=DD, DS=DS, SD=SD, SS=SS, RR=RR)
    else:
        correlation = Count2Correlation(estimator='natural', DD=DD, RR=RR)
    return correlation