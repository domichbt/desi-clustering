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
    # Import cucount submodules for pair counting and particle handling
    from cucount.jax import Particles, BinAttrs, WeightAttrs, setup_logging
    from cucount.jax import create_sharding_mesh
    from cucount.types import count2
    from lsstypes import ObservableLeaf, ObservableTree

    # Use distributed mesh for computation across devices
    with create_sharding_mesh():
        all_fibered_data, all_parent_data = [], []

        # Helper function to extract (RA, DEC) positions and weights from catalog
        def get_rdw(catalog):
            positions = (catalog['RA'], catalog['DEC'])
            # Combine individual weights with optional bitwise weights
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        # Process each data source (for cross-correlations)
        for _get_data in get_data:
            _data = _get_data()
            # Create Particles objects for fibered and parent catalogs in celestial (RA, DEC) coordinates
            fibered_data = Particles(*get_rdw(_data['fibered_data']), positions_type='rd', exchange=True)
            parent_data = Particles(*get_rdw(_data['parent_data']), positions_type='rd', exchange=True)
            all_fibered_data.append(fibered_data)
            all_parent_data.append(parent_data)

        # Define angular separation bins (in degrees)
        theta = 10**np.arange(-5, -1 + 0.1, 0.1)
        battrs = BinAttrs(theta=theta)

        # Set up bitwise weight attributes (PIP weights)
        bitwise = None
        if all_fibered_data[0].get('bitwise_weight'):
            bitwise = dict(weights=all_fibered_data[0].get('bitwise_weight'))
            if jax.process_index() == 0:
                logger.info(f'Applying PIP weights {bitwise}.')

        # Compute pair counts for fibered data with bitwise weights
        wattrs = WeightAttrs(bitwise=bitwise)
        DDfibered = count2(*all_fibered_data, battrs=battrs, wattrs=wattrs)['weight'].value()

        # Compute pair counts for parent (unfiber-limited) data without bitwise weights
        wattrs = WeightAttrs()
        DDparent = count2(*all_parent_data, battrs=battrs, wattrs=wattrs)['weight'].value()

    # Prepare output arrays with angular separation bins
    kw = dict(theta=battrs.coords('theta'), theta_edges=battrs.edges('theta'), coords=['theta'])
    auw = {}
    # Angular upweights = ratio of parent to fibered pair counts (1 where no pairs)
    auw['DD'] = ObservableLeaf(value=np.where(DDfibered == 0., 1., DDparent / DDfibered), **kw)
    #auw['DR'] = ObservableLeaf(value=np.where(DRfibered == 0., 1., DRparent / DRfibered), **kw)

    # Wrap in ObservableTree for consistent data structure
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

    # Set default effective redshift computation parameters
    if zeff is None: zeff = {'boxpad': 1.1, 'cellsize': 10.}
    kw_zeff = dict(zeff)

    # Set up jackknife resampling configuration (default: angular splits)
    if jackknife is None: jackknife = {}
    kw_jackknife = dict(jackknife)
    if kw_jackknife: kw_jackknife = {'mode': 'angular', 'nsplits': 60, 'nside': 512, 'random_state': 42} | kw_jackknife

    # Step 1: Compute effective redshift for clustering measurements
    from .spectrum2_tools import prepare_jaxpower_particles, compute_fkp_effective_redshift
    from jaxpower import create_sharding_mesh

    # Helper to convert single catalog or list to unified format
    def merge_randoms(catalog):
        if not isinstance(catalog, (tuple, list)):
            return catalog
        # Concatenate multiple random catalogs into single object
        return catalog[0].concatenate(catalog)

    # Prepare random catalog loaders for zeff computation
    get_randoms = [lambda: {'randoms': merge_randoms(_get_data_randoms()['randoms'])} for _get_data_randoms in get_data_randoms]

    # Compute effective redshift: density-weighted average of random catalog redshifts
    with create_sharding_mesh(meshsize=kw_zeff.get('meshsize', None)):
        all_particles = prepare_jaxpower_particles(*get_randoms, mattrs=kw_zeff, add_randoms=['IDS'])
        all_randoms = [particles['randoms'] for particles in all_particles]
        # Use object ID as seed for reproducible random splitting
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        # Compute zeff: fraction is normalization of pair counts
        zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, split=seed, resampler='cic', return_fraction=True)
        del all_particles, all_randoms

    # Step 2: Pair counting on distributed mesh
    from cucount.jax import create_sharding_mesh
    with create_sharding_mesh() as sharding_mesh:

        all_data, all_randoms, all_shifted = [], [], []

        # Helper to extract Cartesian positions and weights from catalog
        def get_pw(catalog):
            positions = catalog['POSITION']
            # Combine individual weights with optional bitwise weights
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        # Helper to check if catalog is a list (multiple randoms)
        def _is_list(catalog):
            return isinstance(catalog, (tuple, list))

        # Helper to convert catalog(s) to Particles object(s)
        def get_all_particles(catalog, subsampler=None, as_list=False):
            # If as_list=True, ensure we process as list of catalogs
            if as_list and not _is_list(catalog):
                catalog = [catalog]
            # Recursively convert list of catalogs
            if _is_list(catalog):
                return [get_all_particles(catalog, subsampler=subsampler) for catalog in catalog]  # list of randoms
            positions, weights = get_pw(catalog)
            # Assign jackknife split labels if subsampler provided
            splits = None
            if subsampler is not None:
                splits = subsampler.label(positions).astype('i8')
            # Create Particles with optional jackknife splits
            return Particles(positions, weights=weights, splits=splits, exchange=True)

        # Collect data catalogs for jackknife subsampling setup
        jackknife_particles = []
        for _get_data_randoms in get_data_randoms:
            # Use numpy backend for jackknife particle collection (no JAX)
            data = cucount.numpy.Particles(*get_pw(_get_data_randoms()['data'].gather(mpiroot=None)))
            jackknife_particles.append(data)

        # Set default bin attributes if not provided
        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))

        battrs = BinAttrs(**battrs)

        # Set up theta-cut selection if requested
        sattrs = None
        if cut is not None:
            sattrs = SelectionAttrs(theta=(0., 0.05))
            if jax.process_index() == 0:
                logger.info(f'Applying theta-cut {sattrs}.')

        # Set up bitwise (PIP) and angular weights
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

        # Set up jackknife subsampler if requested
        subsampler = None
        if kw_jackknife:
            from cucount.jax import SplitAttrs
            from cucount.utils import KMeansSubsampler
            # Concatenate all data catalogs for consistent jackknife splitting
            jackknife_particles = cucount.numpy.Particles.concatenate(jackknife_particles)
            # Use K-means to create spatial splits (e.g., angular regions)
            subsampler = KMeansSubsampler(jackknife_particles, wattrs=wattrs, **kw_jackknife)
            spattrs = SplitAttrs(mode='jackknife', nsplits=subsampler.nsplits)
            #labels = subsampler.label(jackknife_particles)
            #print(np.bincount(labels))

        # Process each data source (for stacking multiple catalogs)
        for _get_data_randoms in get_data_randoms:
            # Load data, randoms (optionally shifted) catalogs
            _catalogs = _get_data_randoms()
            # Convert to Particles objects with optional jackknife splits
            data = get_all_particles(_catalogs['data'], subsampler=subsampler)
            randoms = get_all_particles(_catalogs['randoms'], subsampler=subsampler, as_list=True)
            # Shifted randoms used for anisotropic RSD modeling (if provided)
            if _catalogs.get('shifted', None) is not None:
                shifted = get_all_particles(_catalogs['shifted'], subsampler=subsampler, as_list=True)
            else:
                # Default: use same shifted catalogs as randoms
                shifted = [None] * len(randoms)
            all_data.append(data)
            all_randoms.append(randoms)
            all_shifted.append(shifted)

        if jax.process_index() == 0:
            logger.info(f'All particles on the device')

        # Create partial function for pair counting with consistent bin/mesh attributes
        _count2 = partial(count2, battrs=battrs, mattrs=mattrs, sattrs=sattrs, spattrs=spattrs)

        # Compute DD (data-data) pair counts
        DD = _count2(*all_data, wattrs=wattrs)['weight']

        # Clone data, with IIP weights (in case we provided bitwise weights)
        for i in range(len(all_data)):
            all_data[i] = all_data[i].clone(weights=wattrs(all_data[i]))

        # Initialize lists for random and shifted pair counts
        DS, SD, SS, RR = [], [], [], []
        iran = 0

        # Loop over multiple random catalogs (for noise estimation)
        for all_randoms_i, all_shifted_i in zip(zip(*all_randoms, strict=True), zip(*all_shifted, strict=True), strict=True):
            if jax.process_index() == 0:
                logger.info(f'Processing random {iran:d}.')
            iran += 1

            # Compute RR (random-random) pair counts
            RR.append(_count2(*all_randoms_i)['weight'])

            # Compute SS (shifted-shifted) pair counts for anisotropic RSD
            if all(shifted is not None for shifted in all_shifted_i):
                SS.append(_count2(*all_shifted_i)['weight'])
            else:
                # If no shifted catalog, use random as shifted (for isotropic case)
                all_shifted_i = all_randoms_i
                SS.append(RR[-1])

            # Compute DS (data-shifted) and SD (shifted-data) cross counts
            DS.append(_count2(all_data[0], all_shifted_i[-1])['weight'])
            SD.append(_count2(all_shifted_i[0], all_data[-1])['weight'])

    # Sum pair counts across all random catalogs
    DS, SD, SS, RR = (types.sum(XX) for XX in [DS, SD, SS, RR])

    # Create correlation object with Landy-Szalay estimator
    # (handles jackknife covariance if requested, otherwise reduces to standard correlation function)
    correlation = (Count2JackknifeCorrelation if kw_jackknife else Count2Correlation)(estimator='landyszalay', DD=DD, DS=DS, SD=SD, SS=SS, RR=RR)

    # Store effective redshift in metadata
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

    # Use distributed mesh for computation across devices
    with jax.make_mesh((jax.device_count(),), axis_names=('x',), axis_types=(jax.sharding.AxisType.Auto,)):
        all_data, all_shifted = [], []

        # Helper to extract Cartesian positions and weights from catalog
        def get_pw(catalog):
            positions = catalog['POSITION']
            # Combine individual weights with optional bitwise weights
            weights = [catalog['INDWEIGHT']] + _format_bitweights(catalog['BITWEIGHT'] if 'BITWEIGHT' in catalog else None)
            return positions, weights

        # Helper to check if catalog is a list (multiple shifted catalogs)
        def _is_list(catalog):
            return isinstance(catalog, (tuple, list))

        # Helper to convert catalog(s) to Particles object(s)
        def get_all_particles(catalog, as_list=False):
            # If as_list=True, ensure we process as list of catalogs
            if as_list and not _is_list(catalog):
                catalog = [catalog]
            # Recursively convert list of catalogs
            if _is_list(catalog):
                return [get_all_particles(catalog) for catalog in catalog]  # list of shifted catalogs
            positions, weights = get_pw(catalog)
            # Create Particles object and broadcast across devices
            return Particles(positions, weights=weights, exchange=True)

        # Process each data source (for stacking multiple catalogs)
        for _get_data in get_data:
            # Load data (optionally shifted) catalogs
            _catalogs = _get_data()
            # Convert to Particles object
            data = get_all_particles(_catalogs['data'])  # data is not a list of catalogs
            # Shifted catalogs for anisotropic RSD modeling (if provided)
            if _catalogs.get('shifted', None) is not None:
                shifted = get_all_particles(_catalogs['shifted'], as_list=True)
            else:
                # Default: no shifted catalog (isotropic case)
                shifted = [None]
            all_data.append(data)
            all_shifted.append(shifted)
        if jax.process_index() == 0:
            logger.info(f'All particles on the device')

        # Set default bin attributes if not provided
        if battrs is None:
            battrs = dict(s=np.linspace(0., 180., 181), mu=(np.linspace(-1., 1., 201), 'midpoint'))

        battrs = BinAttrs(**battrs)
        wattrs = WeightAttrs()

        # Set up mesh attributes (box size, center) for FFT grid
        mattrs = mattrs or {}
        mattrs = MeshAttrs(*all_data, battrs=battrs, **mattrs)

        # Create partial function for pair counting with consistent attributes
        _count2 = partial(count2, battrs=battrs, mattrs=mattrs)

        # Compute DD (data-data) pair counts
        DD = _count2(*all_data, wattrs=wattrs)['weight']

        # clone data, with individual weights
        for i in range(len(all_data)):
            all_data[i] = all_data[i].clone(weights=wattrs(all_data[i]))

        # Compute RR (random-random) analytically for periodic box (no random catalog needed)
        RR = count2_analytic(battrs=battrs, mattrs=mattrs)

        # Initialize lists for shifted and cross pair counts
        DS, SD, SS = [], [], []
        iran = 0

        # Loop over shifted catalogs (if provided)
        for all_shifted in zip(*all_shifted, strict=True):
            if jax.process_index() == 0:
                logger.info(f'Processing random {iran:d}.')
            # Process only if shifted catalogs are provided
            if all(shifted is not None for shifted in all_shifted):
                # Compute SS (shifted-shifted) pair counts for anisotropic RSD
                SS.append(_count2(*all_shifted)['weight'])
                # Compute DS (data-shifted) and SD (shifted-data) cross counts
                DS.append(_count2(all_data[0], all_shifted[-1])['weight'])
                SD.append(_count2(all_shifted[0], all_data[-1])['weight'])
                iran += 1

    # If we processed shifted catalogs, use Landy-Szalay estimator with all pair counts
    if iran:
        DS, SD, SS = (types.sum(XX) for XX in [DS, SD, SS])
        correlation = Count2Correlation(estimator='landyszalay', DD=DD, DS=DS, SD=SD, SS=SS, RR=RR)
    else:
        # Otherwise, use natural (natural) estimator: 1 + xi = DD / RR
        correlation = Count2Correlation(estimator='natural', DD=DD, RR=RR)
    return correlation