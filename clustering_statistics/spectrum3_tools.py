"""
Fourier-space 3-point clustering measurements.

Main functions
--------------
* `compute_mesh3_spectrum`: Main bispectrum measurement backend.
* `compute_window_mesh3_spectrum`: Compute bispectrum window.
* `compute_box_mesh3_spectrum`: Measure the bispectrum in periodic boxes.
"""

import time
import logging

import numpy as np
import jax
from jax import numpy as jnp
import lsstypes as types

from .tools import compute_fkp_effective_redshift
from .spectrum2_tools import prepare_jaxpower_particles, _get_jaxpower_attrs


logger = logging.getLogger('spectrum3')


def compute_mesh3_spectrum(*get_data_randoms, mattrs=None,
                            basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], edges=None, los='local',
                            buffer_size=0, norm: dict=None, cache=None):
    r"""
    Compute the 3-point spectrum multipoles using mesh-based FKP fields with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return dict of 'data', 'randoms' (optionally 'shifted') catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects. If None, default attributes are used.
        See :func:`prepare_jaxpower_particles` for details.
    basis : str, optional
        Basis for the 3-point spectrum computation. Default is 'sugiyama-diagonal'.
    ells : list of tuples, optional
        List of multipole moments to compute. Default is [(0, 0, 0), (2, 0, 2)] (for the sugiyama basis).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.005 h/\mathrm{Mpc}` is used for the sugiyama basis, :math:`0.01 h/\mathrm{Mpc}` for the scoccimarro basis.
        See :class:`jaxpower.BinMesh3SpectrumPoles` for details.
    los : {'local', 'x', 'y', 'z', array-like}, optional
        Line-of-sight definition. 'local' uses local LOS, 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    buffer_size : int, optional
        Buffer size when binning; if the binning is multidimensional, increase for faster computation at the cost of memory.
    norm : dict, optional
        Optional arguments for computing normalization.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point spectrum multipoles.
    """
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp3_normalization, compute_fkp3_shotnoise, BinMesh3SpectrumPoles, compute_mesh3_spectrum)

    mattrs = mattrs or {}
    # Set up distributed computation mesh across JAX devices
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particle catalogs (data and randoms) with IDS for reproducibility (random splitting based on object IDs)
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        # Attributes about the estimation
        attrs = _get_jaxpower_attrs(*all_particles)
        # Set line-of-sight direction in attributes
        attrs.update(los=los)
        # Get mesh attributes from first particle set (same for all)
        mattrs = all_particles[0]['data'].attrs

        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        # Set default k-space binning step based on basis
        if edges is None: edges = {'step': 0.02 if 'scoccimarro' in basis else 0.005}
        # Set default normalization parameters
        if norm is None: norm = {'cellsize': 10.}
        kw_norm = dict(norm)

        # Check if binning object is cached and still valid for current mesh
        bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            # Create new binning object if not cached or if mesh changed
            bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size)
        # Store binning object in cache for future use
        cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)

        # Create FKP fields from data and random catalogs
        all_fkp = [FKPField(particles['data'], particles['randoms']) for particles in all_particles]
        # Compute FKP normalization: integral of n^3(x)
        # Use IDS for process-invariant random splitting
        norm = compute_fkp3_normalization(*all_fkp, bin=bin, split=[(42, fkp.randoms.extra['IDS']) for fkp in all_fkp],
                                          **kw_norm)

        # Create FKP fields for shot noise computation (using shifted catalogs if available)
        all_fkp = [FKPField(particles['data'], particles['shifted'] if particles.get('shifted', None) is not None else particles['randoms']) for particles in all_particles]
        del all_particles
        # Paint FKP fields onto mesh grid with TSC interpolation and interlacing
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # Compute shot noise: variance of Fourier modes from random particles
        num_shotnoise = compute_fkp3_shotnoise(*all_fkp, los=los, bin=bin, **kw)

        # Wait for all computations to finish before proceeding
        jax.block_until_ready((norm, num_shotnoise))
        if jax.process_index() == 0:
            logger.info('Normalization and shotnoise computation finished')

        # Paint FKP fields onto mesh grids (stored as real-valued arrays to save memory)
        meshes = [fkp.paint(**kw, out='real') for fkp in all_fkp]
        # Clear FKP fields from memory
        del all_fkp

        # JIT-compile spectrum computation for performance (static_argnames for non-JAX arguments)
        jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])

        # Compute bispectrum (3-point spectrum) from mesh grids
        spectrum = jitted_compute_mesh3_spectrum(*meshes, los=los, bin=bin)
        # Attach normalization and shot noise to spectrum object
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        # Propagate coordinate/attribute information to each multipole pole
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        # Attach global attributes to spectrum (z, los, weights, etc.)
        spectrum = spectrum.clone(attrs=attrs)

        # Wait for spectrum computation to complete on all devices
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')

    return spectrum


def _get_window_edges(mattrs, scales: tuple=(1, 4)):
    """Return window edges."""
    # Get maximum separation (1/4 of smallest box dimension)
    distmax, cellmin = mattrs.boxsize.min() / 4., mattrs.cellsize.min()
    # Define cell size progressions: 6 doublings then stop (None)
    nsizes, cellsizes = [6] * 5 + [None], [cellmin * 2**i for i in range(6)]
    edges = []
    # Create edge bins for each scale factor (e.g., 1x and 4x)
    for scale in scales:
        edges_scale = []
        start = 0.
        # Progressively increase cell size with doublings
        for nsize, cellsize in zip(nsizes, cellsizes):
            cellsize = cellsize * scale
            # Use regular spacing up to fixed number or until max distance
            if nsize is None:
                tmp = np.arange(start, distmax * scale / scales[0] + cellsize, cellsize)
            else:
                tmp = start + np.arange(nsize) * cellsize
            # Keep only valid edges and update start point
            if tmp.size:
                start = tmp[-1] + cellsize
                edges_scale.append(tmp)
        # Concatenate all edge segments
        edges_scale = np.concatenate(edges_scale, axis=0)
        # Filter to maximum distance
        edges_scale = edges_scale[edges_scale < distmax * scale / scales[0] + cellsize]
        edges.append(edges_scale)
    return edges


def compute_window_mesh3_spectrum(*get_data_randoms, spectrum, zeff: dict=None, ibatch: tuple=None, computed_batches: list=None, buffer_size=0):
    r"""
    Compute the 3-point spectrum window with :mod:`jaxpower`.

    Parameters
    ----------
    get_data_randoms : callables
        Functions that return tuples of (data, randoms) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    spectrum : Mesh3SpectrumPoles
        Measured 3-point spectrum multipoles.
    zeff : dict, optional
        Optional arguments for computing effective redshift.
        Default is ``{'cellsize': 10.}`` (density computed with ``cellsize = 10.``)
    ibatch : tuple, optional
        To split the window function multipoles to compute in batches, provide (0, nbatches) for the first batch,
        (1, nbatches) for the second, etc; up to (nbatches - 1, nbatches).
        ``None`` to compute the final window matrix.
    computed_batches : list, optional
        The window function multipoles that have been computed thus far.

    Returns
    -------
    spectrum : WindowMatrix or dict of WindowMatrix
        The computed 3-point spectrum window.
    """
    # Import window and correlation functions from jaxpower
    from jaxpower import (create_sharding_mesh, BinMesh3SpectrumPoles, BinMesh3CorrelationPoles, compute_mesh3_correlation,
                           compute_smooth3_spectrum_window, get_smooth3_window_bin_attrs, interpolate_window_function, split_particles)

    # Extract multipole orders from measured spectrum
    ells = spectrum.ells
    # Extract mesh attributes from spectrum (boxsize, gridsize, etc.)
    mattrs = {name: spectrum.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']}
    los = spectrum.attrs['los']
    # Painting parameters: third-order spline with 3-fold interlacing
    kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
    # Set effective redshift computation parameters
    if zeff is None: zeff = {'cellsize': 10.}
    kw_zeff = dict(zeff)

    # Set up distributed computation
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load random catalogs and prepare particles
        all_particles = prepare_jaxpower_particles(*get_data_randoms, mattrs=mattrs, add_randoms=['IDS'])
        all_randoms = [particles['randoms'] for particles in all_particles]
        del all_particles
        # Update mesh attributes from random catalog attributes
        mattrs = all_randoms[0].attrs

        # Extract first multipole pole to get coordinate information
        pole = next(iter(spectrum))
        ells, edges, basis = spectrum.ells, pole.edges('k'), pole.basis
        # Gather normalization from all multipoles
        norm = jnp.concatenate([spectrum.get(ell).values('norm') for ell in spectrum.ells])
        # Get unique k-values and corresponding bin edges
        k, index = np.unique(pole.coords('k', center='mid_if_edges')[..., 0], return_index=True)
        edges = edges[index, 0]
        # Reconstruct edges from bin centers
        edges = np.insert(edges[:, 1], 0, edges[0, 0])
        # Create binning for power spectrum (not bispectrum) correlation
        bin = BinMesh3SpectrumPoles(mattrs, edges=edges, ells=ells, basis=basis, mask_edges='')
        # Maximum k value in binning
        stop = bin.edges1d[0].max()
        # Minimum step between adjacent k bins
        step = np.diff(bin.edges1d[0], axis=-1).min()
        # Create finer binning for input correlation (half step size)
        edgesin = np.arange(0., 1.5 * stop, step / 2.)
        edgesin = jnp.column_stack([edgesin[:-1], edgesin[1:]])

        # Map particle indices to catalog indices (cycle if fewer catalogs than 3 fields)
        fields = list(range(len(all_randoms)))
        fields += [fields[-1]] * (3 - len(all_randoms))
        # Use object IDs for process-invariant random splitting
        seed = [(42, randoms.extra['IDS']) for randoms in all_randoms]
        # Compute effective redshift for window computation (third-order expansion)
        zeff, norm_zeff = compute_fkp_effective_redshift(*all_randoms, order=3, split=seed, return_fraction=True, **kw_zeff)

        correlations = []
        # Get window basis attributes (e.g., which multipoles to compute)
        kw, ellsin = get_smooth3_window_bin_attrs(ells, ellsin=2, fields=fields, return_ellsin=True)
        # Filter to low multipoles only (reduce computational cost)
        kw['ells'] = [ell for ell in kw['ells'] if all(ell <= 2 for ell in ell)]
        # For now, keep only (0,0,0) multipole
        kw['ells'] = kw['ells'][:1]
        # JIT-compile 3-point correlation computation
        jitted_compute_mesh3_correlation = jax.jit(compute_mesh3_correlation, static_argnames=['los'], donate_argnums=[0])

        # Create logarithmic k-grid for window interpolation
        coords = jnp.logspace(-3, 5, 1024)
        # List of scale factors for multigrid computation (coarse and fine)
        list_scales = [1, 4]
        # Get binning edges for each scale
        list_edges = _get_window_edges(mattrs, scales=list_scales)

        ells = kw['ells']
        # If batching, select only subset of multipoles for this batch
        if ibatch is not None:
            start, stop = ibatch[0] * len(ells) // ibatch[1], (ibatch[0] + 1) * len(ells) // ibatch[1]
            kw['ells'] = ells[start:stop]

        # Compute window using multigrid approach if first batch
        if ells and not bool(computed_batches):
            # Painting parameters
            kw_paint = dict(resampler='tsc', interlacing=3, compensate=True)
            # Loop over scale factors (coarse to fine)
            for scale, edges in zip(list_scales, list_edges):
                # Create coarser mesh (larger boxsize)
                mattrs2 = mattrs.clone(boxsize=scale * mattrs.boxsize)
                if jax.process_index() == 0:
                    logger.info(f'Processing scale x{scale:.0f}, using {mattrs2}')
                # Create binning for coarse mesh
                sbin = BinMesh3CorrelationPoles(mattrs2, edges=edges, **kw, buffer_size=buffer_size)

                meshes = []
                # Paint random catalogs on coarse mesh
                for iran, randoms in enumerate(split_particles(all_randoms + [None] * (3 - len(all_randoms)),
                                                               seed=seed, fields=fields)):
                    # Adapt random catalog to coarse mesh and exchange across processes
                    randoms = randoms.clone(attrs=mattrs2).exchange(backend='mpi')
                    # Normalize by data/random weight ratio
                    alpha = pole.attrs['wsum_data'][0][min(iran, len(all_randoms) - 1)] / randoms.weights.sum()
                    # Paint random on mesh scaled by alpha
                    meshes.append(alpha * randoms.paint(**kw_paint, out='real'))

                # Compute 3-point correlation on coarse mesh
                t0 = time.time()
                correlation = jitted_compute_mesh3_correlation(meshes, bin=sbin, los=los)
                # Normalize correlation by average normalization factor
                correlation = correlation.clone(norm=[np.mean(norm)] * len(sbin.ells))
                jax.block_until_ready(correlation)
                if jax.process_index() == 0:
                    logger.info(f"Computed windows {kw['ells']}, scale {scale}, in {time.time() - t0:.2f} s.")
                # Interpolate correlation to fine k-grid
                correlation = interpolate_window_function(correlation.unravel(), coords=coords, order=3)
                correlations.append(correlation)

            # Extract coordinate grids from correlations
            coords = list(next(iter(correlations[0])).coords().values())
            # Create masks for smooth transition between scales (using -3 offset for cubic spline)
            masks = [(coords[0] < edges[-3])[:, None] * (coords[1] < edges[-3])[None, :] for edges in list_edges[:-1]]
            # Add final mask (all points)
            masks.append((coords[0] < np.inf)[:, None] * (coords[1] < np.inf)[None, :])
            weights = []
            for mask in masks:
                if len(weights):
                    # Exclude already-weighted regions
                    weights.append(mask & (~weights[-1]))
                else:
                    weights.append(mask)
            # Regularize weights to avoid zero division
            weights = [np.maximum(mask, 1e-6) for mask in weights]
            # Combine correlations from different scales using weights
            correlation = correlations[0].sum(correlations, weights=weights)

        # If batching, join with previously computed batches
        if computed_batches:
            correlation = types.join(computed_batches)
            # Reorder to match original ells sequence
            correlation = types.join([correlation.get(ells=[ell]) for ell in ells])

        # Wait for window computation to complete
        jax.block_until_ready(correlation)
        if jax.process_index() == 0:
            logger.info('Window functions computed.')

        results = {}
        # Store raw correlation (before binning into spectrum window)
        results['window_mesh3_correlation_raw'] = correlation
        # Build window matrix (final window) if not batching or if all batches computed
        if ibatch is None:
            if jax.process_index() == 0:
                logger.info('Building window matrix.')
            # Convert correlation to window matrix using 2D FFTLog
            window = compute_smooth3_spectrum_window(correlation, edgesin=edgesin, ellsin=ellsin, bin=bin, flags=('fftlog',), batch_size=4)
            # Extract observable from window
            observable = window.observable
            # Update observable with spectrum normalization and zeff attributes
            observable = observable.map(lambda pole, label: pole.clone(norm=spectrum.get(**label).values('norm'), attrs=pole.attrs | dict(zeff=zeff / norm_zeff, norm_zeff=norm_zeff)), input_label=True)
            # Normalize window by average normalization (in case norm is k-dependent)
            window = window.clone(observable=observable, value=window.value() / (norm[..., None] / np.mean(norm)))
            results['raw'] = window
    return results


def compute_box_mesh3_spectrum(*get_data, mattrs=None,
                                basis='sugiyama-diagonal', ells=[(0, 0, 0), (2, 0, 2)], edges=None, los='z',
                                buffer_size=0, mask_edges=None, cache=None):
    r"""
    Compute the 3-point spectrum multipoles for a cubic box using :mod:`jaxpower`.

    Parameters
    ----------
    get_data : callables
        Functions that return tuples of (data, [shifted]) catalogs.
        See :func:`prepare_jaxpower_particles` for details.
    mattrs : dict, optional
        Mesh attributes to define the :class:`jaxpower.ParticleField` objects.
        See :func:`prepare_jaxpower_particles` for details.
    ells : list of int, optional
        List of multipole moments to compute. Default is (0, 2, 4).
    edges : dict, optional
        Edges for the binning; array or dictionary with keys 'start' (minimum :math:`k`), 'stop' (maximum :math:`k`), 'step' (:math:`\Delta k`).
        If ``None``, default step of :math:`0.001 h/\mathrm{Mpc}` is used.
        See :class:`jaxpower.BinMesh3SpectrumPoles` for details.
    los : {'x', 'y', 'z', array-like}, optional
        Line-of-sight direction. If 'x', 'y', 'z' use fixed axes, or provide a 3-vector.
    cache : dict, optional
        Cache to store binning class (can be reused if ``meshsize`` and ``boxsize`` are the same).
        If ``None``, a new cache is created.

    Returns
    -------
    spectrum : Mesh3SpectrumPoles
        The computed 3-point spectrum multipoles.
    """
    from jaxpower import (create_sharding_mesh, FKPField, compute_fkp3_shotnoise, compute_box3_normalization, BinMesh3SpectrumPoles, compute_mesh3_spectrum, compute_fkp3_shotnoise)

    # Use provided mesh attributes or empty dict
    mattrs = mattrs or {}
    # Set up distributed computation mesh
    with create_sharding_mesh(meshsize=mattrs.get('meshsize', None)):
        # Load and prepare particle catalogs (data and optional shifted)
        all_particles = prepare_jaxpower_particles(*get_data, mattrs=mattrs)
        # Extract JAX power attributes from particles
        attrs = _get_jaxpower_attrs(*all_particles)
        # Set line-of-sight direction
        attrs.update(los=los)
        # Get mesh attributes from first particle set
        mattrs = all_particles[0]['data'].attrs

        # Initialize or retrieve cached binning object
        if cache is None: cache = {}
        # Retrieve cached binning or None if not cached
        bin = cache.get(f'bin_mesh3_spectrum_{basis}', None)
        # Set default k-space binning (finer for sugiyama)
        if edges is None: edges = {'step': 0.02 if 'scoccimarro' in basis else 0.005}

        # Create binning if not cached or if mesh parameters changed
        if bin is None or not np.all(bin.mattrs.meshsize == mattrs.meshsize) or not np.allclose(bin.mattrs.boxsize, mattrs.boxsize):
            bin = BinMesh3SpectrumPoles(mattrs, edges=edges, basis=basis, ells=ells, buffer_size=buffer_size, mask_edges=mask_edges)
        # Cache the binning object
        cache.setdefault(f'bin_mesh3_spectrum_{basis}', bin)

        # Extract data catalogs (periodic box has no randoms)
        all_data = [particles['data'] for particles in all_particles]
        # Compute normalization for periodic box (simpler than survey case)
        norm = compute_box3_normalization(*all_data, bin=bin)

        # Create FKP fields for shot noise (use shifted if available, else use data)
        all_fkp = [FKPField(particles['data'], particles['shifted']) if particles.get('shifted', None) is not None else particles['data'] for particles in all_particles]
        # Clear particle data from memory
        del all_particles
        # Compute shot noise for periodic box
        num_shotnoise = compute_fkp3_shotnoise(*all_fkp, bin=bin, fields=None)

        # Painting parameters: TSC with interlacing
        kw = dict(resampler='tsc', interlacing=3, compensate=True)
        # Paint all FKP fields onto mesh (store as real arrays to save memory)
        all_mesh = []
        for fkp in all_fkp:
            mesh = fkp.paint(**kw, out='real')
            # Subtract mean
            all_mesh.append(mesh - mesh.mean())
        # Clear FKP fields from memory
        del all_fkp

        # JIT-compile spectrum computation for performance
        jitted_compute_mesh3_spectrum = jax.jit(compute_mesh3_spectrum, static_argnames=['los'])
        # Compute bispectrum from mesh grids
        spectrum = jitted_compute_mesh3_spectrum(*all_mesh, bin=bin, los=los)
        # Attach normalization and shot noise
        spectrum = spectrum.clone(norm=norm, num_shotnoise=num_shotnoise)
        # Propagate attributes to each multipole pole
        spectrum = spectrum.map(lambda pole: pole.clone(attrs=attrs))
        # Attach global attributes to spectrum
        spectrum = spectrum.clone(attrs=attrs)
        # Wait for computation to complete on all devices
        jax.block_until_ready(spectrum)
        if jax.process_index() == 0:
            logger.info('Mesh-based computation finished')
    return spectrum