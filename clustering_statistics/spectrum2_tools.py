"""
Fourier-space 2-point clustering measurements.

Main functions
--------------
* `prepare_jaxpower_particles`: Convert catalogs into mesh-ready particle inputs.
* `compute_mesh2_spectrum`: Main `P(k)` measurement backend.
* `compute_window_mesh2_spectrum`: Compute the power spectrum window matrix.
* `compute_window_mesh2_spectrum_fm`: Build forward-model window matrix.
* `compute_covariance_mesh2_spectrum`: Estimate Fourier-space covariance.
* `run_preliminary_fit_mesh2_spectrum`: Run preliminary fits used in covariance matrix.
"""

import logging
from collections.abc import Callable

import numpy as np
import jax
from jax import numpy as jnp
import lsstypes as types

from .tools import default_mpicomm, _format_bitweights, compute_fkp_effective_redshift, combine_stats


logger = logging.getLogger('spectrum2')


@default_mpicomm
def prepare_jaxpower_particles(*get_data_randoms, mattrs=None, add_data=tuple(), add_randoms=tuple(), **kwargs):
    """
    Prepare :class:`jaxpower.ParticleField` objects from data and randoms catalogs.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' (optionally 'randoms', 'shifted') catalogs.
        Each catalog must contain 'POSITION' and 'INDWEIGHT', and optionally 'BITWEIGHT' for bitwise weights and 'TARGETID'
        for randoms IDs to allow process-invariant random split in bispectrum normalization.
    mattrs : dict, optional
        Mesh attributes ('boxsize', 'meshsize' or 'cellsize', 'boxcenter') to define the :class:`ParticleField` objects. If ``None``, default attributes are used.
    kwargs : dict, optional
        Additional keyword arguments to pass to :class:`ParticleField`.

    Returns
    -------
    all_particles : list of dictionaries
        List of dictionaries of :class:`ParticleField`  'data' (optionally 'randoms', 'shifted') objects for each input catalog.
    """
    # Import mesh attribute computation and particle field creation from jaxpower
    from jaxpower.mesh import get_mesh_attrs, ParticleField
    # Use MPI backend for distributed particle processing across processes
    backend = 'mpi'
    # Extract MPI communicator from kwargs (added by @default_mpicomm decorator)
    mpicomm = kwargs['mpicomm']

    # Load all catalogs by calling the provided functions
    all_catalogs = [_get_data_randoms() for _get_data_randoms in get_data_randoms]

    # Define the mesh attributes; pass in positions only
    # check=True validates that all positions are within mesh bounds
    mattrs = get_mesh_attrs(*[catalog['POSITION'] for catalogs in all_catalogs for catalog in catalogs.values()], check=True, **(mattrs or {}))
    if jax.process_index() == 0:
        logger.info(f'Using mesh {mattrs}.')

    # Use IDS instead
    def collective_arange(local_size):
        # Compute global array indices across all MPI processes
        # This allows each process to know its global position in the distributed array
        sizes = mpicomm.allgather(local_size)
        return sum(sizes[:mpicomm.rank]) + np.arange(local_size)

    all_particles = []
    # Dictionary mapping 'data' and 'randoms' to their respective extra columns to load
    add = {'data': add_data, 'randoms': add_randoms}
    for catalogs in all_catalogs:
        particles = {}
        for name, catalog in catalogs.items():
            extra = {}
            # Start with individual weights from catalog
            indweights = catalog['INDWEIGHT']
            if name == 'data':
                # Extract and process bitwise weights (fiber weights, completeness, etc.)
                bitweights = None
                if 'BITWEIGHT' in catalog and 'BITWEIGHT' in add[name]:
                    # Parse bitwise weight array into individual weight components
                    bitweights = _format_bitweights(catalog['BITWEIGHT'])
                    from cucount.jax import BitwiseWeight
                    # Compute individual inverse probability weight (IIP) from bitwise components
                    # p_correction_nbits=False: no impact on IIP computation
                    iip = BitwiseWeight(weights=bitweights, p_correction_nbits=False)(bitweights)
                    # Store original bitweights in extra
                    extra['BITWEIGHT'] = [indweights] + bitweights
                    # Multiply individual weights by IIP to correct fiber assignment at large scales
                    indweights = indweights * iip
                # Add any additional columns (e.g., Z, WEIGHT_FKP) to extra dictionary
                for column in add[name]:
                    if column != 'BITWEIGHT': extra[column] = catalog[column]
            elif name == 'randoms':
                # Extract target IDs from random catalog for reproducible random splitting
                if 'TARGETID' in catalog and 'IDS' in add[name]:
                    extra['IDS'] = catalog['TARGETID']
                # Add other requested columns to extra dictionary
                for column in add[name]:
                    if column != 'IDS': extra[column] = catalog[column]
            # Create ParticleField object: positions + weights + mesh attributes
            # exchange=True: distribute particles across MPI processes by spatial location
            # This ensures load balancing across processes
            particle = ParticleField(catalog["POSITION"], indweights, attrs=mattrs, exchange=True, backend=backend, extra=extra, **kwargs)
            particles[name] = particle
        all_particles.append(particles)
    if jax.process_index() == 0:
        logger.info(f'All particles on the device')

    return all_particles


def _get_jaxpower_attrs(*all_particles):
    """Return summary attributes from :class:`jaxpower.ParticleField` objects: total weight and size."""
    # Get mesh attributes from first particle set (same for all)
    mattrs = next(iter(all_particles[0].values())).attrs
    # Creating FKP fields
    attrs = {}
    for particles in all_particles:
        for name in particles:
            if particles[name] is not None:
                # Store total weight sum for each particle type
                if f'wsum_{name}' not in attrs:
                    #attrs[f'size_{name}'] = [[]]  # size is process-dependent
                    attrs[f'wsum_{name}'] = [[]]
                #attrs[f'size_{name}'][0].append(particles[name].size)
                # Sum weights across all processes using MPI
                attrs[f'wsum_{name}'][0].append(particles[name].sum())
    # Extract and preserve mesh geometric information for output
    for name in ['boxsize', 'boxcenter', 'meshsize']:
        attrs[name] = mattrs[name]
    return attrs


def compute_mesh2_spectrum(*get_data_randoms, mattrs=None, cut=None, auw=None,
                           ells=(0, 2, 4), edges=None, los='firstpoint', optimal_weights=None,
                           norm: dict=None, cache=None):
    r"""
    Compute the 2-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.
    cut : bool, optional
        If True, apply a theta-cut of (0, 0.05) in degrees.
    auw : ObservableTree, optional
        Angular upweights to apply. If ``None``, no angular upweights are applied.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'local', 'firstpoint', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'firstpoint' uses the position of the first point in the pair,
        'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    norm : dict, optional
        Optional arguments for computing normalization.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles or dict of Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles. If `cut` or `auw` are provided, returns a dict with keys 'raw', 'cut', and/or 'auw'.
    """

    # Import FKP field, power spectrum computation, and binning tools from jaxpower
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp2_normalization, compute_fkp2_shotnoise, BinMesh2SpectrumPoles, compute_mesh2_spectrum,
                          BinParticle2SpectrumPoles, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise)

    # Collect column names needed for optimal weight computation
    columns_optimal_weights = []
    if optimal_weights is not None:
        # Get required columns (default: Z for redshift-dependent weights)
        columns_optimal_weights += getattr(optimal_weights, 'columns', ['Z'])   # to compute optimal weights, e.g. for fnl
    mattrs = mattrs or {}
    # Set up distributed mesh computation across JAX devices (multi-GPU/CPU)
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load particles and prepare for FKP field creation
        # add_data=['BITWEIGHT'] for fiber collision corrections
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_data=['BITWEIGHT'] + columns_optimal_weights, add_randoms=columns_optimal_weights)

        # Initialize or retrieve cached binning object from previous runs
        if cache is None: cache = {}
        # Set default k-space binning step (0.001 h/Mpc)
        if edges is None: edges = {'step': 0.001}
        # Set default normalization computation parameters (density from 10 Mpc/h cells)
        if norm is None: norm = {'cellsize': 10.}
        kw_norm = dict(norm)

        def _compute_spectrum_ell(all_particles, ells, fields=None):
            # Compute power spectrum for input given multipoles
            # Gather particle attributes (weights, mesh info) for output metadata
            attrs = _get_jaxpower_attrs(*all_particles)
            # Store line-of-sight direction in attributes
            attrs.update(los=los)
            # Get mesh attributes from first data particle
            mattrs = all_particles[0]['data'].attrs

            # Define the binner for k-space binning
            key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
            bin = cache.get(key, None)
            # Create new binning if not cached or if mesh parameters changed
            if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
                bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
            # Store binning in cache for future use
            cache.setdefault(key, bin)

            all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
            # Computing normalization: integral of density^2, splitting randoms ('split') to avoid common noise
            norm = compute_fkp2_normalization(*all_fkp, bin=bin, **kw_norm)

            # Computing shot noise from shifted catalogs (reconstruction) or use randoms if no shifted available
            all_fkp = [FKPField(particles['data'], particles['shifted'] if particles.get('shifted', None) is not None else particles['randoms']) for particles in all_particles]
            del all_particles
            # Shot noise computed from (shifted) randoms
            num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=fields)

            # Wait for normalization and shot noise to complete on all devices
            jax.block_until_ready((norm, num_shotnoise))
            if jax.process_index() == 0:
                logger.info('Normalization and shotnoise computation finished')

            results = {}
            # First compute the theta-cut (close-pair) contribution for contamination correction
            if cut is not None:
                # Define angular selection: only pairs separated by < 0.05 degrees
                sattrs = {'theta': (0., 0.05)}
                #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, ells=ells)
                # Use correlation binning for close pairs (finer radial bins for accuracy)
                pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, ells=ells)
                from jaxpower.particle2 import convert_particles
                # Convert FKP fields to particle pairs for direct pair counting
                all_particles = [convert_particles(fkp.particles) for fkp in all_fkp]
                # Count close pairs directly (no mesh needed, exact calculation)
                close = compute_particle2(*all_particles, bin=pbin, los=los)
                # Attach normalization and shot noise, then convert to power spectrum
                close = close.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields), norm=norm)
                # Convert correlation poles to power spectrum (multiply by bin centers)
                close = close.to_spectrum(bin.xavg)
                # Store negative contribution (contamination to subtract)
                results['cut'] = -close.value()

            # Then compute the AUW-weighted (angular upweight) pairs and bitwise-weighted pairs
            with_bitweights = 'BITWEIGHT' in all_fkp[0].data.extra
            if auw is not None or with_bitweights:
                from cucount.jax import WeightAttrs
                from jaxpower.particle2 import convert_particles
                # Define angular selection for close pairs (< 0.1 degrees for bitwise weights)
                sattrs = {'theta': (0., 0.1)}
                bitwise = angular = None
                if with_bitweights:
                    # Reconstruct weights for fiber collision corrections
                    all_data = [convert_particles(fkp.data, weights=list(fkp.data.extra['BITWEIGHT']) + [fkp.data.weights], exchange_weights=False) for fkp in all_fkp]
                    # Extract bitwise weight structure (sets nrealizations based on BITWEIGHT size, fine to use the first)
                    bitwise = dict(weights=all_data[0].get('bitwise_weight'))
                    if jax.process_index() == 0:
                        logger.info(f'Applying PIP weights {bitwise}.')
                else:
                    # No bitwise weights, remove individual weights from AUW * individual_weight
                    all_data = [convert_particles(fkp.data, weights=[fkp.data.weights] * 2, exchange_weights=False, index_value=dict(individual_weight=1, negative_weight=1)) for fkp in all_fkp]
                # Apply angular upweights if provided (fiber collision corrections)
                if auw is not None:
                    # Extract angular separation and weight values from pre-computed AUW
                    angular = dict(sep=auw.get('DD').coords('theta'), weight=auw.get('DD').value())
                    if jax.process_index() == 0:
                        logger.info(f'Applying AUW {angular}.')
                # Set up weight attributes for pair counting
                wattrs = WeightAttrs(bitwise=bitwise, angular=angular)
                # Create binning for close-pair data-data counts with weights
                pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, wattrs=wattrs, ells=ells)
                # Count weighted pairs directly
                DD = compute_particle2(*all_data, bin=pbin, los=los)
                # Attach normalization and shot noise
                DD = DD.clone(num_shotnoise=compute_particle2_shotnoise(*all_data, bin=pbin, fields=fields), norm=norm)
                results['auw'] = DD.value()

            # Wait for particle-based calculations to complete
            jax.block_until_ready(results)
            if jax.process_index() == 0:
                logger.info(f'Particle-based calculation finished')

            # Paint particles onto mesh grids for Fourier-space power spectrum computation
            kw = dict(resampler='tsc', interlacing=3, compensate=True)
            # out='real' to save memory (store as real arrays instead of complex)
            all_mesh = [fkp.paint(**kw, out='real') for fkp in all_fkp]
            # Free memory from FKP fields
            del all_fkp

            # JIT the mesh-based spectrum computation; helps with memory footprint
            jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
            #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
            # Compute power spectrum from painted mesh grids via FFT
            spectrum = jitted_compute_mesh2_spectrum(*all_mesh, bin=bin, los=los)
            # Attach normalization and shot noise to spectrum
            spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
            # Propagate particle attributes to each multipole
            spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
            # Also attach attributes at spectrum level
            spectrum = spectrum.clone(attrs=attrs)
            # Wait for spectrum computation to complete on all devices
            jax.block_until_ready(spectrum)
            if jax.process_index() == 0:
                logger.info('Mesh-based computation finished')

            # Add theta-cut and AUW contributions to the base spectrum
            for name, value in results.items():
                # Combine contamination corrections with raw spectrum
                results[name] = spectrum.clone(value=spectrum.value() + value)
            # Store raw spectrum without contamination corrections
            results['raw'] = spectrum

            return results

        # Compute power spectrum either without or with optimal weights
        if optimal_weights is None:
            # Standard case: use FKP weights, compute all ells at once
            results = _compute_spectrum_ell(all_particles, ells=ells)
        else:
            # Optimal weights case: compute ell-by-ell due to dependence of optimal weight on multipole
            # Prepare fields tuple for multi-tracer support (pad to length 2)
            fields = tuple(range(len(all_particles)))
            # Pad to length 2 if we have fewer catalogs (for cross-correlation compatibility)
            fields = fields + (fields[-1],) * (2 - len(fields))
            # Pad particle list similarly for processing
            all_particles = tuple(all_particles) + (all_particles[-1],) * (2 - len(all_particles))
            results = {}
            # Loop over each multipole moment
            for ell in ells:
                if jax.process_index() == 0:
                    logger.info(f'Applying optimal weights for ell = {ell:d}')

                def _get_optimal_weights(all_particles):
                    # Generator that yields particles with optimal weights applied for this ell
                    # all_particles is [data1, data2] or [randoms1, randoms2] or [shifted1, shifted2]
                    if all_particles[0] is None:  # shifted is None, yield None
                        while True:
                            yield tuple(None for particles in all_particles)
                    # Get optimal weights from the weight function for all particles
                    for all_weights in optimal_weights(ell, [{'INDWEIGHT': particles.weights} | {column: particles.extra[column] for column in columns_optimal_weights} for particles in all_particles]):
                        # Yield particles with weights replaced by optimal weights
                        yield tuple(particles.clone(weights=weights) for particles, weights in zip(all_particles, all_weights))

                result_ell = {}
                # Get names of particle types (data, randoms, shifted)
                names = list(all_particles[0].keys())
                # Loop over all weight combinations for this ell
                for _all_particles in zip(*[_get_optimal_weights([particles[name] for particles in all_particles]) for name in names]):
                    # _all_particles is a list [(data1, data2), (randoms1, randoms2), [(shifted1, shifted2)]] of tuples of ParticleField with optimal weights applied
                    # Reorder to group by catalog index rather than particle type
                    _all_particles = list(zip(*_all_particles))
                    # _all_particles is now a list of tuples [(data1, randoms1, shifted1), (data2, randoms2, shifted2)] with optimal weights applied
                    # Convert tuples back to dictionaries for _compute_spectrum_ell
                    _all_particles = [dict(zip(names, _particles)) for _particles in _all_particles]
                    # _all_particles is now a list of dictionaries [{'data': data1, 'randoms': randoms1, 'shifted': shifted1}, {'data': data2, 'randoms': randoms2, 'shifted': shifted2}] with optimal weights applied
                    # Compute spectrum for this ell and weight combination
                    _result = _compute_spectrum_ell(_all_particles, ells=[ell], fields=fields)
                    # Collect results for all variants (raw, cut, auw)
                    for key in _result:  # raw, cut, auw
                        result_ell.setdefault(key, [])
                        result_ell[key].append(_result[key])
                # Average over weight combinations (if multiple tracers: sum over 1<->2 cross-weights)
                for key, value in result_ell.items():
                    results.setdefault(key, [])
                    # Combine_stats sums over different weights
                    results[key].append(combine_stats(value))  # sum 1<->2
            # Combine all ells into single observable tree structure
            for key in results:
                # types.join concatenates along multipole axis
                results[key] = types.join(results[key])  # join multipoles

    # Return single result or dictionary of variants
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def compute_window_mesh2_spectrum(*get_data_randoms, spectrum: types.Mesh2SpectrumPoles, optimal_weights: Callable=None, cut
: bool=None, zeff: dict=None):
    r"""
    Compute the 2-point spectrum window with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.
    optimal_weights : callable or None, optional
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    zeff : dict, optional
        Optional arguments for computing effective redshift.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)

    Returns
    -------
    window : WindowMatrix or dict of WindowMatrix
        The computed 2-point spectrum window. If `auw` is provided, returns a dict with keys 'raw' and 'auw'.
    """
    # FIXME: data is not used, could be dropped, add auw
    # Import window and correlation computation tools from jaxpower
    from jaxpower import (create_sharding_mesh, BinMesh2SpectrumPoles, BinMesh2CorrelationPoles, compute_mesh2_correlation, BinParticle2CorrelationPoles, compute_particle2, compute_particle2_shotnoise,
                           compute_smooth2_spectrum_window, get_smooth2_window_bin_attrs, interpolate_window_function, split_particles)

    # Extract multipole moments from input spectrum
    ells = spectrum.ells
    # Extract mesh parameters from spectrum attributes
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    # Extract line-of-sight direction
    los = spectrum.attrs['los']
    # Theory multipoles for window computation (fixed basis for window calculation)
    ellsin = [0, 2, 4]
    # Mesh painting parameters: TSC kernel with 3-fold interlacing for aliasing correction
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
    # Set default effective redshift computation parameters
    if zeff is None: zeff = {'cellsize': 10.}
    kw_zeff = dict(zeff)

    # Collect column names needed for optimal weight computation
    columns_optimal_weights = []
    if optimal_weights is not None:
        # Get required columns (default: Z for redshift-dependent weights)
        columns_optimal_weights += getattr(optimal_weights, 'columns', ['Z'])   # to compute optimal weights, e.g. for fnl
    mattrs = mattrs or {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load random catalogs and prepare particles with IDS for reproducible splitting
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'] + columns_optimal_weights)
        # Extract only randoms (data not used in window computation)
        all_randoms = [particles['randoms'] for particles in all_particles]
        del all_particles

        # Determine k-space binning edges from input spectrum
        stop, step = -np.inf, np.inf
        for pole in spectrum:
            # Get k-space edges from each pole
            edges = pole.edges('k')
            # Find maximum k and minimum step across all poles
            stop = max(edges.max(), stop)
            step = min(np.nanmin(np.diff(edges, axis=-1)), step)
        # Create finer k-binning for theory
        edgesin = np.arange(0., 1.2 * stop, step)
        # Convert to column-stacked format [k_min, k_max] for each bin
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

        def _compute_window_ell(all_randoms, ells, isum=0, fields=None):
            # Compute window function for specified multipole moments
            all_randoms = list(all_randoms)
            # Use IDS (TARGETID) from random catalog for process-invariant random splitting
            seed = [(42, randoms.extra["IDS"]) for randoms in all_randoms]  # for process invariance
            mattrs = all_randoms[0].attrs
            # Get spectrum pole at first ell to extract edges
            pole = spectrum.get(ells[0])
            # Create binning object for window function
            bin = BinMesh2SpectrumPoles(mattrs, edges=pole.edges('k'), ells=ells)
            # Get normalization from input power spectrum
            norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in ells], axis=0)
            results = {}
            correlations = []
            # Get window basis attributes (which multipoles to compute in correlation space)
            kw_window = get_smooth2_window_bin_attrs(ells, ellsin)
            # JIT-compile correlation computation for memory efficiency
            # donate_argnums=[0] allows JAX to reuse memory of first argument
            jitted_compute_mesh2_correlation = jax.jit(compute_mesh2_correlation, static_argnames=['los'], donate_argnums=[0])
            # Window computed in configuration space, summing Bessel functions over the Fourier-space mesh
            # Use logarithmic s-grid for robust interpolation
            coords = jnp.logspace(-3, 5, 4 * 1024)
            list_edges = []
            # Loop over scale factors for multigrid window computation (coarse then fine)
            for scale in [1, 4]:
                # Create coarser mesh (larger boxsize) for computational efficiency at coarse scales
                mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
                if jax.process_index() == 0:
                    logger.info(f'Processing scale x{scale:.0f}, using {mattrs2}')
                all_mesh = []
                # Paint random catalogs on coarse mesh
                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (2 - len(all_randoms)), seed=seed, fields=fields)):
                    # Redistribute particles to coarse mesh and exchange across MPI processes
                    randoms = randoms.clone(attrs=mattrs2).exchange(backend='mpi')
                    # Compute weight normalization (data/random density ratio from input spectrum)
                    alpha = pole.attrs['wsum_data'][isum][min(iran, len(all_randoms) - 1)] / randoms.weights.sum()
                    # Paint random particles with proper normalization onto mesh
                    all_mesh.append(alpha * randoms.paint(**kw_paint, out='real'))
                # Define radial binning for correlation space window
                # distmax: use 1/4 of boxsize minimum dimension for correlation range
                distmax, cellsize = mattrs2.boxsize.min() / 4., mattrs2.cellsize.min()
                # Create radial bins from 0 to distmax
                edges = np.arange(0., distmax + cellsize, cellsize)
                list_edges.append(edges)
                # Create binning for correlation function (configuration space)
                sbin = BinMesh2CorrelationPoles(mattrs2, edges=edges, **kw_window, basis='bessel')
                # Compute correlation function via FFT (inverse Fourier transform of painted mesh)
                correlation = jitted_compute_mesh2_correlation(all_mesh, bin=sbin, los=los).clone(norm=[np.mean(norm)] * len(sbin.ells))
                # Free mesh memory
                del all_mesh
                #if jax.process_index() == 0: correlation.write(f'_tests/window_correlation2_{scale:.0f}.h5')
                # Interpolate correlation to fine logarithmic k-grid for FFTLog integration
                correlation = interpolate_window_function(correlation, coords=coords, order=3)
                correlations.append(correlation)
            # Create transition masks between coarse and fine scale grids
            # Masks ensure each point is covered by exactly one scale
            masks = [coords < edges[-3] for edges in list_edges[:-1]]
            # Last mask covers remainder
            masks.append((coords < np.inf))
            # Convert masks to exclusive regions (each point weighted by only one scale)
            weights = []
            for mask in masks:
                if len(weights):
                    # Exclude already-weighted regions from previous scales
                    weights.append(mask & (~weights[-1]))
                else:
                    weights.append(mask)
            # Regularize weights to avoid division by zero
            weights = [np.maximum(mask, 1e-6) for mask in weights]
            # Combine correlations from different scales using smooth weights
            results['window_mesh2_correlation_raw'] = correlation = correlations[0].sum(correlations, weights=weights)

            # Convert correlation to power spectrum window via FFTLog
            window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
            # Update window with spectrum normalization from input power spectrum
            observable = window.observable.map(lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'), attrs=pole.attrs), input_label=True)
            # Normalize window by average normalization (in case norm is k-dependent)
            # Division corrects for any k-dependent normalization variation
            results['raw'] = window.clone(observable=observable, value=window.value() / (norm[..., None] / np.mean(norm)))  # just in case norm is k-dependent
            if cut:
                # Compute theta-cut contribution to window
                sattrs = {'theta': (0., 0.05)}
                #pbin = BinParticle2SpectrumPoles(mattrs, edges=bin.edges, xavg=bin.xavg, sattrs=sattrs, **kw_window)
                # Use correlation binning for close pairs (finer bins than spectrum)
                pbin = BinParticle2CorrelationPoles(mattrs, edges={'step': 0.1}, sattrs=sattrs, **kw_window)
                from jaxpower.particle2 import convert_particles
                all_particles = []
                # Convert randoms to particles and apply weight normalization
                for iran, randoms in enumerate(all_randoms):
                    alpha = pole.attrs['wsum_data'][isum][iran] / randoms.weights.sum()
                    all_particles.append(convert_particles(randoms.clone(weights=alpha * randoms.weights)))
                # Count close pairs directly
                correlation = compute_particle2(*all_particles, bin=pbin, los=los)
                # Attach normalization and shot noise
                correlation = correlation.clone(num_shotnoise=compute_particle2_shotnoise(*all_particles, bin=pbin, fields=fields), norm=[np.mean(norm)] * len(sbin.ells))
                pole = next(iter(correlation))
                # Interpolate close pair correlation to logarithmic grid
                correlation = interpolate_window_function(correlation, coords=coords, order=3)
                results['window_mesh2_correlation_cut'] = correlation
                # Combine raw and close-pair correlations (add contributions)
                correlation = correlation.clone(value=results['window_mesh2_correlation_raw'].value() + correlation.value())
                # Convert combined correlation to power spectrum window
                window = compute_smooth2_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',))
                results['cut'] = window.clone(observable=results['raw'].observable, value=window.value() / (norm[..., None] / np.mean(norm)))
            # Convert correlation results to observable tree structure (for consistency with other outputs)
            for key, result in results.items():
                if 'correlation' in key:
                    # Wrap correlation in ObservableTree with appropriate ell structure
                    results[key] = types.ObservableTree([result], oells=[ells[0] if len(ells) == 1 else tuple(ells)])
            return results

        # Compute window either without or with optimal weights
        if optimal_weights is None:
            # Standard case: use FKP weights, compute all ells at once
            # Compute effective redshift for window computation
            fields = None
            seed = [(42, randoms.extra["IDS"]) for randoms in all_randoms]
            # Compute zeff and normalization fraction (for reporting)
            zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=2, split=seed, fields=fields, return_fraction=True, **kw_zeff)
            results = _compute_window_ell(all_randoms, ells=ells, fields=fields)
            # Attach zeff to window attributes
            for key in results:
                if 'correlation' not in key:
                    observable = results[key].observable
                    # Add effective redshift to each pole's attributes
                    observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)))
                    results[key] = results[key].clone(observable=observable)
        else:
            # Optimal weights case: compute ell-by-ell due to weight dependence on multipole
            results = {}
            # Prepare fields tuple for multi-tracer support (pad to length 2)
            fields = tuple(range(len(all_randoms)))
            fields = fields + (fields[-1],) * (2 - len(fields))
            # Pad random particle list similarly
            all_randoms = tuple(all_randoms) + (all_randoms[-1],) * (2 - len(all_randoms))
            # Loop over multipoles
            for ell in ells:
                if jax.process_index() == 0:
                    logger.info(f'Applying optimal weights for ell = {ell:d}')

                def _get_optimal_weights(all_particles):
                    # Generator for optimal weights applied to particles
                    # all_particles is [data1, data2] or [randoms1, randoms2] or [shifted1, shifted2]
                    if all_particles[0] is None:  # shifted is None, yield None
                        while True:
                            yield tuple(None for particles in all_particles)
                    def clone(particles, weights):
                        # Clone particle with new weights
                        toret = particles.clone(weights=weights)
                        return toret

                    # Get optimal weights for this ell from the weight function
                    for all_weights in optimal_weights(ell, [{"INDWEIGHT": particles.weights} | {column: particles.extra[column] for column in columns_optimal_weights} for particles in all_particles]):
                        yield tuple(clone(particles, weights=weights) for particles, weights in zip(all_particles, all_weights))

                result_ell = {}
                # Loop over weight combinations for this ell
                for isum, _all_randoms in enumerate(_get_optimal_weights(all_randoms)):
                    # Loop over weight combinations for the same multipole
                    fields = None
                    seed = [(42, randoms.extra["IDS"]) for randoms in _all_randoms]
                    # Compute zeff for this weight combination
                    zeff, norm_zeff = compute_fkp_effective_redshift(*_all_randoms, order=2, split=seed, fields=fields, return_fraction=True, **kw_zeff)
                    _result = _compute_window_ell(_all_randoms, ells=[ell], isum=isum, fields=fields)
                    # Attach zeff to output for each weight combination
                    for key in _result:  # raw, cut, auw
                        if 'correlation' not in key:
                            observable = _result[key].observable
                            observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)))
                            _result[key] = _result[key].clone(observable=observable)
                        result_ell.setdefault(key, [])
                        result_ell[key].append(_result[key])
                # Average over weight combinations
                for key, windows in result_ell.items():
                    results.setdefault(key, [])
                    # windows can be WindowMatrix and ObservableTree (window correlation)
                    window = combine_stats(windows)  # sum 1<->2
                    # Used power spectrum norm is for the sum of the two;
                    # just sum the two components
                    window = window.clone(value=sum(window.value() for window in windows))
                    results[key].append(window)
            # Combine results into final observable structure
            for key in results:
                if 'correlation' in key:
                    # Join correlations along multipole axis
                    results[key] = types.join(results[key])
                else:
                    # Join windows along multipole axis
                    observables = [window.observable for window in results[key]]
                    observable = types.join(observables)
                    value = np.concatenate([window.value() for window in results[key]], axis=0)
                    results[key] = results[key][0].clone(value=value, observable=observable)  # join multipoles

    return results


def compute_window_mesh2_spectrum_fm(
    *get_data_randoms: Callable,
    spectrum: types.Mesh2SpectrumPoles,
    theory: types.Mesh2SpectrumPoles,
    optimal_weights: Callable | None,
    data_to_randoms_ratio: float,
    catalog_split_seed: int,
    geo: bool,
    ric_nbins: int,
    ric_regions: list[str],
    amr: bool,  # is optional
    regression_maps: list[str] | None,
    templates_paths_kwargs: dict | None,
    amr_regions_zranges: list[tuple[str, tuple[float, float]]] | None,
    spectrum_regions: list[str] | None,
    unitary_amplitude: bool = True,
    n_realizations: int,
    seeds: list[int] | None,
    batch_size: int = 4,
) -> dict[str, dict[str, list[types.WindowMatrix]]]:
    """
    Compute the 2-point spectrum window with :mod:`desiwinds`.

    Parameters
    ----------
    *get_data_randoms : Callable
        Functions that return tuples of (data, randoms) catalogs.
    spectrum : lsstypes.Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles. Only used for their attributes, not their values.
    theory: lsstypes.Mesh2SpectrumPoles
        Input theory power spectrum, used as a fiducial for the derivative. Attributes (e.g. ells) used for mock survey generation; value used for the derivative.
    optimal_weights : Callable or None
        Function taking (ell, catalog) as input and returning total weights to apply to data and randoms.
        It can have an optional attribute 'columns' that specifies which additional columns are needed to compute the optimal weights.
        As a default, ``optimal_weights.columns = ['Z']`` to indicate that redshift information is needed.
        A dictionary ``catalog`` of columns is provided, containing 'INDWEIGHT' and the requested columns.
        If ``None``, no optimal weights are applied.
    data_to_randoms_ratio : float
        Population ratio between "data" and "randoms" to pick in the input randoms catalogs. Must be between 0 and 1.
    catalog_split_seed : int
        Random seed to use for the random split between "data" and "randoms" in the input randoms catalogs.
    geo : bool
        Whether to return the sampled window for the geometry. If False, only the RIC (±AMR) contribution is returned.
    ric_nbins : int
        Number of radial bins to use for the RIC.
    ric_regions : list[str]
        Regions to use for the RIC, e.g. ``["N", "S"]`` or ``["N", "SnoDES", "DES]``.
    amr : bool
        Whether to apply the angular mode removal (AMR), i.e. to forward model the power loss due to linear angular systematics weights.
    regression_maps : list[str] | None
        Names of the systematics templates to use for the AMR. Can be set to ``None`` if ``amr=False``.
    templates_paths_kwargs : dict
        Keyword arguments to pass to the function loading the templates maps, e.g. paths to the templates files, EBV map, nside, etc. Not needed if ``amr=False``. Must at least contain the keys ``templates_path_N`` and ``templates_path_S`` with the paths to the templates files for the Northern and Southern regions, respectively.
    amr_regions_zranges : list[tuple[str, tuple[float, float]]] | None
        Regions where to apply the regressions for the AMR, and corresponding redshift ranges. Can be set to ``None`` if ``amr=False``.
    spectrum_regions : list[str] | None
        Regions for which to compute the window and power spectrum. If ``None``, the whole catalog is used as one region. Typically ``["NGC", "SGC"]``.
    n_realizations : int
        Number of realizations to compute.
    seeds : list[int] | None
        Seeds to use for each realization. If ``None``, defaults to ``2 * i_realization + 3``.
    unitary_amplitude : bool, optional
        Whether to use unitary amplitude for the mock survey mesh generation, by default True.
    batch_size : int, optional
        Number of window computations to run in parallel, by default 4. Depends on the available memory, number of randoms catalogs, size of the mesh... Lower if needed.

    Returns
    -------
    dict[str, dict[str, list[lsstypes.WindowMatrix]]]
        Dictionary, per effect included (geometry, RIC, RIC+AMR) and per region, of lists of window matrices (one per realization).
    """
    assert len(seeds) == n_realizations if seeds is not None else True, "If seeds are provided, their number must match n_realizations."
    # Notes to self:
    # * RIC not optional
    # * n_randoms is effectively set by the length of get_data_randoms
    import mpytools as mpy
    from desiwinds.forward import mock_survey_catalog, prepare_AMR, prepare_RIC
    from desiwinds.window import get_window_spikes
    from jaxpower import BinMesh2SpectrumPoles, FKPField, ParticleField, compute_fkp2_normalization, create_sharding_mesh

    from .tools import add_photometric_template_values, select_region

    def _add_photometric_template_values(catalogs: dict[str, mpy.Catalog]):
        return {name: add_photometric_template_values(catalogs[name], regression_maps, **templates_paths_kwargs) for name in catalogs}

    def _select_region(catalogs: dict[str, mpy.Catalog], spectrum_region: str) -> dict[str, mpy.Catalog]:
        return {name: catalog[select_region(ra=catalog["RA"], dec=catalog["DEC"], region=spectrum_region)] for name, catalog in catalogs.items()}

    def _split_data_randoms(catalogs: dict[str, mpy.Catalog]) -> dict[str, mpy.Catalog]:
        """Split the randoms into "data" and "randoms" based on the provided ratio. Overwrite original "data"."""
        data_size = int(data_to_randoms_ratio * catalogs["randoms"].size)  # MPI local
        randoms_size = catalogs["randoms"].size - data_size
        rng = mpy.random.MPIRandomState(seed=catalog_split_seed, size=catalogs["randoms"].size)  # Use local sizes
        mask_is_data = rng.uniform() < (data_size / (data_size + randoms_size))
        data = catalogs["randoms"][mask_is_data]
        randoms = catalogs["randoms"][~mask_is_data]
        return {"data": data, "randoms": randoms}

    def _update_fkp(data_weights, randoms_weights, fkp_field, estimator_weights):
        return fkp_field.clone(
            data=fkp_field.data.clone(
                weights=data_weights * getattr(fkp_field.data, estimator_weights, 1.0),
            ),
            randoms=fkp_field.randoms.clone(
                weights=randoms_weights * getattr(fkp_field.randoms, estimator_weights, 1.0),
            ),
        )

    def _safe_divide(a, b):
        return jnp.where(b != 0, a / b, 0.0)

    spectrum_regions = spectrum_regions or []
    columns_optimal_weights = []
    if optimal_weights is not None:
        columns_optimal_weights += getattr(optimal_weights, "columns", [])  # to compute optimal weights, e.g. for fnl

    # Recover output and mesh information from the observable spectrum
    ellsout = spectrum.ells
    los = spectrum.attrs["los"]  # this has to match with theory input
    if los in ["endpoint", "firstpoint"]:
        los = "local"
    mattrs = {name: spectrum.attrs[name] for name in ["boxsize", "boxcenter", "meshsize"]}

    with create_sharding_mesh(meshsize=mattrs.get("meshsize", None)):
        # Split into "data" and randoms based on the provided ratio
        def wrap(f):
            return lambda: _split_data_randoms(f())

        get_data_randoms = [wrap(_get_data_randoms) for _get_data_randoms in get_data_randoms]

        if amr:  # Add photometric template values to the catalogs, if AMR is applied, as they are needed for the regression

            def wrap(f):
                return lambda: _add_photometric_template_values(f())

            get_data_randoms = [wrap(_get_data_randoms) for _get_data_randoms in get_data_randoms]

        if len(spectrum_regions) > 0:  # Split catalogs into pk regions, if specified

            def wrap(f, spectrum_region):
                return lambda: _select_region(f(), spectrum_region)

            get_data_randoms = [
                wrap(_get_data_randoms, spectrum_region) for spectrum_region in spectrum_regions for _get_data_randoms in get_data_randoms
            ]  # [func1_region1, func2_region1, func3_region1 ... func1_region2, func2_region2, func3_region2 ...]

        all_particles = prepare_jaxpower_particles(
            *get_data_randoms,
            mattrs=mattrs,
            add_randoms=["IDS", "WEIGHT_FKP", "Z", *regression_maps, *columns_optimal_weights],
            add_data=["WEIGHT_FKP", "Z", *regression_maps, *columns_optimal_weights],
        )
        all_randoms = [particles["randoms"] for particles in all_particles]
        all_data = [particles["data"] for particles in all_particles]
        del all_particles

        # Make into len(spectrum_regions) catalogs if split into spectrum regions, otherwise one catalog
        nregion = len(spectrum_regions) if len(spectrum_regions) > 0 else 1
        nrandoms = len(all_randoms)
        chunk_size = nrandoms // nregion
        all_randoms = [ParticleField.concatenate(all_randoms[chunk_size * i : chunk_size * (i + 1)]) for i in range(nregion)]
        all_data = [ParticleField.concatenate(all_data[chunk_size * i : chunk_size * (i + 1)]) for i in range(nregion)]

        for iregion in range(nregion):
            # Randoms
            extra = all_randoms[iregion].extra
            if amr:
                template_values = jnp.stack([extra.pop(map_name) for map_name in regression_maps], axis=-1)
                extra.update({"template_values": template_values})
            # extra already has weight_FKP, just remove from weights=indweights which contains FKP weights
            all_randoms[iregion] = all_randoms[iregion].clone(
                extra=extra, weights=_safe_divide(all_randoms[iregion].weights, all_randoms[iregion].extra["WEIGHT_FKP"])
            )

            # Data
            extra = all_data[iregion].extra
            if amr:
                template_values = jnp.stack([extra.pop(map_name) for map_name in regression_maps], axis=-1)
                extra.update({"template_values": template_values})
            all_data[iregion] = all_data[iregion].clone(extra=extra, weights=_safe_divide(all_data[iregion].weights, all_data[iregion].extra["WEIGHT_FKP"]))
        del extra

        if jax.process_index() == 0:
            logger.info("Catalogs ready, starting preparation...")

        # Prepare arguments for the window computation function
        ric_args = prepare_RIC(data=all_data, randoms=all_randoms, regions=ric_regions, n_bins=ric_nbins, apply_to="randoms")

        if amr:
            extra_effects = "RIC+AMR"
            amr_args = prepare_AMR(data=all_data, randoms=all_randoms, regions_zranges=amr_regions_zranges, apply_to="randoms")
            for iregion in range(nregion):
                extra = all_randoms[iregion].extra
                del extra["template_values"]
                all_randoms[iregion] = all_randoms[iregion].clone(extra=extra)
                # data
                extra = all_data[iregion].extra
                del extra["template_values"]
                all_data[iregion] = all_data[iregion].clone(extra=extra)
            del extra
        else:
            extra_effects = "RIC"
            amr_args = None

        # Turn into FKP fields
        fkp_fields = [FKPField(data=d, randoms=r, attrs=mattrs) for d, r in zip(all_data, all_randoms, strict=True)]
        del all_data, all_randoms
        # Compute FKP normalization for each region, with the estimator weights, and for each ell if optimal weights are applied
        if optimal_weights is None:
            if jax.process_index() == 0:
                logger.info("Using FKP weights, computing window for all ells at once.")
            # Using FKP weights which are symetrical, so this remains an autocorr
            binner = BinMesh2SpectrumPoles(fkp_fields[0].attrs, edges=spectrum.get(0).edges("k"), ells=ellsout)  # TODO: check edges are ok

            # Temporarily add FKP weights to the fkp_fields weights for norm and analytical computation
            fkp_norms = [
                compute_fkp2_normalization(_update_fkp(fkp.data.weights, fkp.randoms.weights, fkp, "WEIGHT_FKP"), bin=binner, cellsize=10.0)
                for fkp in fkp_fields
            ]

            ## FM based computations
            windows = {}

            # Shared window FM arguments
            window_fm_kw = {
                "mock_survey": mock_survey_catalog,
                "theory": theory,
                "nreal": n_realizations,
                "seeds": seeds,
                "batch_size": batch_size,
                "mock_survey_args": (*fkp_fields,),
                "static_argnames": ["los", "unitary_amplitude", "estimator_weights"],
                "tmpdir": None,  # No temporary output
                "survey_names": spectrum_regions,
            }
            mock_survey_kwargs = {
                "los": los,
                "unitary_amplitude": unitary_amplitude,
                "nam_args": None,
                "fkp_norms": fkp_norms,
                "binner": binner,
                "estimator_weights": "WEIGHT_FKP",
                "data_regions": ric_args.data_regions,
                "randoms_regions": ric_args.randoms_regions,
            }

            if geo:
                if jax.process_index() == 0:
                    logger.info("Computing geometry window with desiwinds...")
                _, windows_fm_geo = get_window_spikes(
                    **window_fm_kw,
                    mock_survey_kwargs=mock_survey_kwargs | {"ric_args": None, "amr_args": None},
                )

                windows["geometry"] = windows_fm_geo

            if jax.process_index() == 0:
                logger.info("Computing total window with desiwinds...")
            _, windows_fm = get_window_spikes(
                **window_fm_kw,
                mock_survey_kwargs=mock_survey_kwargs | {"ric_args": ric_args, "amr_args": amr_args},
            )

            windows[extra_effects] = windows_fm
            if jax.process_index() == 0:
                logger.info("desiwinds window computation finished.")

            return windows

        else:
            if jax.process_index() == 0:
                logger.info("Using optimal weights, computing windows for each ell separately.")
            # Optimal weights: non symmetrical, so need to compute "cross-correlation" (same tracer, different weights) + not the same for all ells
            # Proceed ell per ell and sum the windows at the end
            def _attach_weights(fkp_field, ell):
                data_w1, data_w2 = next(
                    optimal_weights(
                        ell,
                        [
                            {column: fkp_field.data.extra[column] for column in ["Z", *columns_optimal_weights]}
                            | {"INDWEIGHT": fkp_field.data.weights * fkp_field.data.extra["WEIGHT_FKP"]}
                        ],
                    )
                )

                randoms_w1, randoms_w2 = next(
                    optimal_weights(
                        ell,
                        [
                            {column: fkp_field.randoms.extra[column] for column in ["Z", *columns_optimal_weights]}
                            | {"INDWEIGHT": fkp_field.randoms.weights * fkp_field.randoms.extra["WEIGHT_FKP"]}
                        ],
                    )
                )
                # These weights also contain real weights and FKP weights ; need to remove the real weights to isolate the "estimator weights" to apply at computation time in the FM
                return fkp_field.clone(
                    data=fkp_field.data.clone(
                        extra=fkp_field.data.extra
                        | {
                            "weight_optimal_1": _safe_divide(data_w1, fkp_field.data.weights),
                            "weight_optimal_2": _safe_divide(data_w2, fkp_field.data.weights),
                        }
                    ),
                    randoms=fkp_field.randoms.clone(
                        extra=fkp_field.randoms.extra
                        | {
                            "weight_optimal_1": _safe_divide(randoms_w1, fkp_field.randoms.weights),
                            "weight_optimal_2": _safe_divide(randoms_w2, fkp_field.randoms.weights),
                        }
                    ),
                )

            windows = {extra_effects: {}}
            if geo:
                windows["geometry"] = {}

            for ell in ellsout:
                binner = BinMesh2SpectrumPoles(fkp_fields[0].attrs, edges=spectrum.get(ell).edges("k"), ells=[ell])  # TODO: check edges are ok
                fkp_fields = [_attach_weights(fkp_field, ell) for fkp_field in fkp_fields]
                # Compute FKP normalization for each region, with the estimator weights (cross correlation), and for given ell = binner
                fkp_norms = [
                    compute_fkp2_normalization(
                        _update_fkp(fkp.data.weights, fkp.randoms.weights, fkp, "weight_optimal_1"),
                        _update_fkp(fkp.data.weights, fkp.randoms.weights, fkp, "weight_optimal_2"),
                        bin=binner,
                        cellsize=10.0,
                    )
                    for fkp in fkp_fields
                ]

                # Shared window FM arguments
                window_fm_kw = {
                    "mock_survey": mock_survey_catalog,
                    "theory": theory,
                    "nreal": n_realizations,
                    "seeds": seeds,
                    "batch_size": batch_size,
                    "mock_survey_args": [(fkp,) * 2 for fkp in fkp_fields],  # same FKP field but with different weights
                    "static_argnames": ["los", "unitary_amplitude", "estimator_weights"],
                    "tmpdir": None,  # No temporary output
                    "survey_names": spectrum_regions,
                }
                mock_survey_kwargs = {
                    "los": los,
                    "unitary_amplitude": unitary_amplitude,
                    "nam_args": None,
                    "fkp_norms": fkp_norms,
                    "binner": binner,  # one ell only
                    "estimator_weights": ("weight_optimal_1", "weight_optimal_2"),
                    "data_regions": ric_args.data_regions,
                    "randoms_regions": ric_args.randoms_regions,
                }

                if geo:
                    if jax.process_index() == 0:
                        logger.info("Computing geometry window for ell=%i with desiwinds...", ell)
                    _, _windows_fm_geo = get_window_spikes(
                        **window_fm_kw,
                        mock_survey_kwargs=mock_survey_kwargs | {"ric_args": None, "amr_args": None},
                    )

                    windows["geometry"][ell] = _windows_fm_geo

                if jax.process_index() == 0:
                    logger.info("Computing total window for ell=%i with desiwinds...", ell)
                _, _windows_fm = get_window_spikes(
                    **window_fm_kw,
                    mock_survey_kwargs=mock_survey_kwargs | {"ric_args": (ric_args,) * 2, "amr_args": (amr_args,) * 2},
                )

                windows[extra_effects][ell] = _windows_fm

            if jax.process_index() == 0:
                logger.info("desiwinds window computation finished.")

            # For each region, sum the windows over ells and apply control variate

            def _combine_ells(windows):
                observables = [window.observable for window in windows]
                observable = types.join(observables)
                value = np.concatenate([window.value() for window in windows], axis=0)
                return windows[0].clone(value=value, observable=observable)  # join multipoles

            if geo:
                windows["geometry"] = {
                    spectrum_region: [_combine_ells([windows["geometry"][ell][ireal][idx] for ell in ellsout]) for ireal in range(n_realizations)]
                    for idx, spectrum_region in enumerate(spectrum_regions)
                }

            windows[extra_effects] = {
                spectrum_region: [_combine_ells([windows[extra_effects][ell][ireal][idx] for ell in ellsout]) for ireal in range(n_realizations)]
                for idx, spectrum_region in enumerate(spectrum_regions)
            }

            return windows


def run_preliminary_fit_mesh2_spectrum(data: types.Mesh2SpectrumPoles, window: types.WindowMatrix, select: dict=None, theory: str='rept', fixed=tuple(), out: types.Mesh2SpectrumPoles=None):
    """
    Compute a smooth theory spectrum to assume when building the covariance.

    Parameters
    ----------
    data : Mesh2SpectrumPoles or None
        Measured spectrum multipoles used to build the covariance and (optionally)
        to set priors / initialize the fit. If None, the function will still
        construct an analytic covariance from `window` but cannot use data-driven
        priors.
    window : WindowMatrix
        Window matrix describing mode-coupling of the estimator. The window's
        observable axes are matched to the `data` before fitting.
    select : dict, optional
        If provided, a selection is applied to `data` via `data.select(**select)`
        prior to fitting (e.g. to restrict k-ranges or multipoles).
    theory : str, optional
        Theory to use in the fit, one of ['rept', 'kaiser'].
    out : Mesh2SpectrumPoles, optional
        If provided, returns a clone of these power spectrum multipoles with best fit theory values.

    Returns
    -------
    out : Mesh2SpectrumPoles
    """
    # Import spectrum computation and covariance tools from jaxpower
    from jaxpower import MeshAttrs, compute_spectrum2_covariance
    # Select k-range for fitting (avoid very low k)
    smooth = data.select(k=(0.001, 10.))

    # Create mesh attributes from input data
    mattrs = MeshAttrs(**{name: data.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    # Compute Gaussian covariance (assuming Gaussian density field)
    covariance = compute_spectrum2_covariance(mattrs, data)  # Gaussian, diagonal covariance

    # Apply selection to data (restrict to fitting range)
    select = select or {'k': (0.02, 10.)}
    data = data.select(**select)
    # Match window to data range
    window = window.at.observable.match(data)
    # Restrict window theory to coverage of measurement
    window = window.at.theory.select(k=(0.001, 1.2 * next(iter(data)).coords('k').max()))
    # Match covariance to data range
    covariance = covariance.at.observable.match(data)
    # Extract effective redshift from window
    z = window.observable.get(ells=0).attrs['zeff']

    import numpy as np
    # FIXME
    np.trapz = np.trapezoid

    # Import clustering theory classes from desilike
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles, REPTVelocileptorsTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    # Select theory model (Kaiser or REPT with velocileptors)
    Theory = {'rept': REPTVelocileptorsTracerPowerSpectrumMultipoles, 'kaiser': KaiserTracerPowerSpectrumMultipoles}[theory]

    # Create fiducial theory template at measurement redshift
    template = FixedPowerSpectrumTemplate(fiducial='DESI', z=z)
    # Instantiate theory model with template
    theory = Theory(template=template)
    # Create observable: combines data, theory, and window function
    observable = TracerPowerSpectrumMultipolesObservable(data=data, window=window, theory=theory)
    # Create likelihood: Gaussian likelihood with computed covariance
    likelihood = ObservablesGaussianLikelihood(observable, covariance=covariance.value())
    # Fix specified parameters
    for param in fixed:
        likelihood.all_params[param].update(fixed=True)

    # Minimize likelihood to get best-fit theory
    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    # Get best-fit parameters
    params = profiles.bestfit.choice(index='argmax', input=True)
    if out is None:
        # Build smooth theory spectrum from best-fit parameters
        poles = []
        for ill, ell in enumerate(theory.ells):
            if ell in smooth.ells:
                # Use original smooth data as template
                pole = smooth.get(ells=ell)
            else:
                # Create new pole with zero shot noise for missing multipoles
                pole = smooth.get(ells=0).clone(meta={"ell": ell})
                if ell != 0:
                    # Zero out shot noise for higher multipoles (only monopole has shot noise)
                    pole = pole.clone(num_shotnoise=np.zeros_like(pole.values("num_shotnoise")))
            # Evaluate theory at k-values from data
            theory.init.update(k=pole.coords("k"))
            value = theory(**params)[ill]
            pole = pole.clone(value=value)
            poles.append(pole)
        # Build spectrum object from theory poles
        smooth = types.Mesh2SpectrumPoles(poles, attrs=smooth.attrs)
    else:
        # Use provided output structure
        value = []
        for label, pole in out.items(level=1):
            # Evaluate theory at k-values
            theory.init.update(k=pole.coords('k'))
            value.append(theory(**params)[theory.ells.index(label['ells'])])
        smooth = out.clone(value=value)
    return smooth


def compute_covariance_mesh2_spectrum(*get_data_randoms, theory=None, fields=None, mattrs=None):
    r"""
    Compute the 2-point spectrum covariance with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data' and 'randoms' catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    theory : Mesh2SpectrumPoles
        Theory 2-point spectrum multipoles.
    fields : tuple, list, optional
        Field names.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.

    Returns
    -------
    covarance : CovarianceMatrix
        The computed 2-point spectrum covariance.
    """
    # Import covariance and window computation tools from jaxpower
    from jaxpower import create_sharding_mesh, compute_fkp2_covariance_window, interpolate_window_function, compute_spectrum2_covariance, FKPField
    # Use FFTLog for reliable correlation-to-spectrum conversion
    fftlog = True
    # Use default fields (1, 2, ...) if not provided
    if fields is None:
        fields = list(range(1, 1 + len(get_data_randoms)))

    results = {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particles
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        # Create FKP fields for covariance window computation
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        mattrs = all_fkp[0].attrs
        # Set correlation binning parameters (finer than spectrum binning)
        kw = dict(edges={'step': mattrs.cellsize.min()}, basis='bessel') if fftlog else dict(edges={})
        # Add fields for cross-covariance and random splitting seed
        kw.update(los='local', fields=fields, split=[(42, fkp.randoms.extra['IDS']) for fkp in all_fkp])
        # Mesh painting parameters: TSC with interlacing
        kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
        # Compute covariance window function (correlation in configuration space)
        windows = compute_fkp2_covariance_window(all_fkp, **kw, **kw_paint)
        #if jax.process_index() == 0: windows.write(f'_tests/window_correlation.h5')
        if fftlog:
            # Very robust to this choice of FFTLog grid
            # Use logarithmic s-grid for interpolation
            coords = np.logspace(-2, 8, 8 * 1024)
            # Interpolate window functions to fine s-grid
            windows = windows.map(lambda window: interpolate_window_function(window, coords=coords), level=1)
        # Store raw correlation windows for diagnostics
        results['window_covariance_mesh2_correlation'] = windows

    # Convert correlation to power spectrum covariance matrix via FFTLog
    covariance = compute_spectrum2_covariance(windows, theory, flags=['smooth'] + (['fftlog'] if fftlog else []))
    # Update label names to match observable structure
    fields = covariance.observable.fields
    # Create observable tree with proper labels
    observable = types.ObservableTree(list(covariance.observable), observables=['spectrum2'] * len(fields), tracers=fields)
    covariance = covariance.clone(observable=observable)
    # Store in results dict
    results['raw'] = covariance
    return results


def compute_rotation_mesh2_spectrum(window: types.WindowMatrix, covariance: types.CovarianceMatrix, Minit: str='momt',
                                    data: types.Mesh2SpectrumPoles=None, theory: types.Mesh2SpectrumPoles=None, select: dict=None):
    """
    Compute the rotation to make the window matrix more diagonal.

    Parameters
    ----------
    window : WindowMatrix
        Window matrix.
    covariance : CovarianceMatrix
        Covariance of the measured spectrum.
    Minit : {'momt', ...}, optional
        Initialization method passed to rotation.setup(Minit=...). Defaults to 'momt'.
    data : Mesh2SpectrumPoles or None, optional
        Measured spectrum used to set priors for the rotation (if available).
    theory : Mesh2SpectrumPoles or None, optional
        Theory spectrum used together with `data` when setting priors.

    Returns
    -------
    rotation : WindowRotationSpectrum2
    """
    # Import rotation matrix computation from jaxpower
    from jaxpower import WindowRotationSpectrum2
    # Extract observable from window or data
    observable = window.observable
    if data is not None:
        # Use data as observable instead of window
        if select is not None:
            data = data.select(**select)
        observable = data
    # Match window observable to target observable (reorder/subset as needed)
    window = window.at.observable.match(observable)
    if theory is not None:
        def interpolate_pole(ref, pole):
            # Interpolate theory to match reference k-values
            return ref.clone(value=np.interp(ref.coords('k'), pole.coords('k'), pole.value()))

        # Interpolate theory to window theory k-values
        theory = window.theory.map(lambda pole, label: interpolate_pole(pole, theory.get(ells=label['ells'])), input_label=True, level=1)
    # Match covariance observable to target observable
    covariance = covariance.at.observable.match(observable)
    # Create rotation matrix object
    rotation = WindowRotationSpectrum2(window=window, covariance=covariance, xpivot=0.1)
    # Set up rotation matrix (initialize using 'momt' method: moment-based initialization)
    rotation.setup(Minit=Minit)
    # Fit rotation matrix to data (if provided)
    rotation.fit()
    if rotation.with_momt and data is not None:
        # To set up priors for rotation parameters from data
        rotation.set_prior(data=data, theory=theory)
    return rotation


def compute_box_mesh2_spectrum(*get_data, ells=(0, 2, 4), edges=None, los='z', cache=None, mattrs=None):
    r"""
    Compute the 2-point spectrum multipoles for a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callables
        Functions that return dict of 'data' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh2SpectrumPoles` for details.
    los : {'x', 'y', 'z', array-like}, optional
        Line-of-sight direction. If 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh2SpectrumPoles
        The computed 2-point spectrum multipoles.
    """
    # Import tools for periodic box power spectrum computation
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp2_shotnoise, compute_box2_normalization, BinMesh2SpectrumPoles, compute_mesh2_spectrum, compute_fkp2_shotnoise)

    mattrs = mattrs or {}
    # Set up distributed computation across JAX devices
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particles (data + optional shifted for RSD distortions)
        all_particles = prepare_jaxpower_particles(*get_data, mattrs=mattrs)
        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        # Set default k-space binning step (0.001 h/Mpc)
        if edges is None: edges = {'step': 0.001}
        # Gather particle attributes (weights, mesh info)
        attrs = _get_jaxpower_attrs(*all_particles)
        # Store line-of-sight direction
        attrs.update(los=los)
        mattrs = all_particles[0]['data'].attrs

        # Define the binner for k-space binning
        key = 'bin_mesh2_spectrum_{}'.format('_'.join(map(str, ells)))
        bin = cache.get(key, None)
        # Create new binning if not cached or mesh changed
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh2SpectrumPoles(mattrs, edges=edges, ells=ells)
        # Store binning in cache
        cache.setdefault(key, bin)

        # Computing normalization for periodic box (simpler than survey: no randoms)
        all_data = [particles['data'] for particles in all_particles]
        norm = compute_box2_normalization(*all_data, bin=bin)

        # Computing shot noise from shifted or data catalogs
        # shifted if reconstruction
        all_fkp = [FKPField(particles['data'], particles['shifted']) if particles.get('shifted', None) is not None else particles['data'] for particles in all_particles]
        # Free memory
        del all_particles
        num_shotnoise = compute_fkp2_shotnoise(*all_fkp, bin=bin, fields=None)

        # Paint particles on mesh grid
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # out='real' to save memory (store as real arrays)
        all_mesh = []
        for fkp in all_fkp:
            # Paint particle density on mesh
            mesh = fkp.paint(**kw, out='real')
            all_mesh.append(mesh - mesh.mean())
        # Free FKP field memory
        del all_fkp
        # JIT the mesh-based spectrum computation; helps with memory footprint
        jitted_compute_mesh2_spectrum = jax.jit(compute_mesh2_spectrum, static_argnames=['los'])
        #jitted_compute_mesh2_spectrum = compute_mesh2_spectrum
        # Compute power spectrum from painted meshes via FFT
        spectrum = jitted_compute_mesh2_spectrum(*all_mesh, bin=bin, los=los)
        # Attach normalization and shot noise
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        # Propagate attributes to output
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        spectrum = spectrum.clone(attrs=attrs)
        # Wait for computation to complete
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')
    return spectrum


def compute_window_box_mesh2_spectrum(spectrum: types.Mesh2SpectrumPoles, zsnap: float=None):
    r"""
    Compute the 2-point spectrum window for a box (i.e., binning window) with :mod:`jaxpower`.

    Parameters
    ----------
    spectrum : Mesh2SpectrumPoles
        Measured 2-point spectrum multipoles.

    Returns
    -------
    window : WindowMatrix
        The computed 2-point spectrum window.
    """
    # Compute binning window for periodic box power spectrum
    from jaxpower import create_sharding_mesh, MeshAttrs, BinMesh2SpectrumPoles, compute_mesh2_spectrum_window

    # Extract mesh attributes and multipoles from input spectrum
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    ells = spectrum.ells
    pole = spectrum.get(0)
    # Set up distributed computation
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        # Create binning from spectrum edges
        bin = BinMesh2SpectrumPoles(mattrs, edges=pole.edges('k'), ells=ells)
        #edgesin = np.linspace(bin.edges.min(), bin.edges.max(), 2 * (len(bin.edges) - 1))
        # Use input spectrum bins as theory bins
        edgesin = bin.edges
        # For box, window is just the binning matrix (no mode coupling from survey effects)
        window = compute_mesh2_spectrum_window(mattrs, edgesin=edgesin, ellsin=ells, los=los, bin=bin)
        observable = window.observable
        # Attach redshift/snapshot information if provided
        if zsnap is not None:
            observable = observable.map(lambda pole: pole.clone(attrs=pole.attrs | dict(zeff=zsnap, zsnap=zsnap)))
        window = window.clone(observable=observable)
    return window


def compute_covariance_box_mesh2_spectrum(theory: types.Mesh2SpectrumPoles=None, mattrs=None):
    r"""
    Compute the 2-point spectrum covariance for a box with :mod:`jaxpower`.

    Parameters
    ----------
    theory : Mesh2SpectrumPoles, optional
        Theory spectrum used together with `spectrum` when setting priors.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects,
        'boxsize', 'meshsize' or 'cellsize', 'boxcenter'. If ``None``, default attributes are used.

    Returns
    -------
    covarance : CovarianceMatrix
        The computed 2-point spectrum covariance.
    """
    # Compute Gaussian covariance for periodic box power spectrum
    from jaxpower import create_sharding_mesh, MeshAttrs, compute_spectrum2_covariance
    # Add zero shot noise to theory for covariance computation
    theory_sn = theory.map(lambda pole: pole.clone(num_shotnoise=pole.values('num_shotnoise') * 0.), level=2)
    mattrs = mattrs or {}
    # Set up distributed computation
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        mattrs = MeshAttrs(**mattrs)
        # Compute Gaussian, diagonal covariance
        covariance = compute_spectrum2_covariance(mattrs, theory_sn)  # Gaussian, diagonal covariance

        # Update label names to match observable structure
        fields = covariance.observable.fields
        # Create observable tree with proper labels
        observable = types.ObservableTree(list(covariance.observable), observables=['spectrum2'] * len(fields), tracers=fields)
        covariance = covariance.clone(observable=observable)
    return covariance