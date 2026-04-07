"""
High-level orchestration for cutsky clustering measurements.

This module provides the main CLI entry point (`clustering-stats`) and the
pipeline driver used for DESI lightcone clustering statistics.

Main functions
--------------
* `compute_stats_from_options`, which takes as input a list of summary statistics to compute and a dictionary of options,
and orchestrates the workflow:
- fill fiducial defaults
- read clustering catalogs and randoms
- optionally run reconstruction
- dispatch to statistic-specific backends, such as `compute_mesh2_spectrum` for power spectrum measurement or `compute_particle2_correlation` for correlation function measurement.
* `postprocess_stats_from_options`, which can be used to run postprocessing steps,
such as combining measurements from different regions or computing rotation matrices for the power spectrum.
"""

import os
import logging
import functools
import copy
import warnings
from pathlib import Path
import itertools

import numpy as np
import jax
import jax.experimental.multihost_utils
import lsstypes as types

from . import tools
from .tools import fill_fiducial_options, _merge_options, Catalog, setup_logging
from .correlation2_tools import compute_angular_upweights, compute_particle2_correlation
from .spectrum2_tools import (
    compute_mesh2_spectrum,
    compute_window_mesh2_spectrum,
    compute_covariance_mesh2_spectrum,
    run_preliminary_fit_mesh2_spectrum,
    compute_rotation_mesh2_spectrum,
    compute_window_mesh2_spectrum_fm,
)
from .spectrum3_tools import compute_mesh3_spectrum, compute_window_mesh3_spectrum
from .recon_tools import compute_reconstruction


logger = logging.getLogger('summary-statistics')


def _expand_cut_auw_options(stat, options):
    # Helper to generate separate option dictionaries for raw, theta-cut, and angular upweight variants
    # For spectrum measurements, create variants with different options
    if 'spectrum' in stat:
        keys = ['cut', 'auw']
        kw = dict(options)
        for key in keys: kw.pop(key, None)
        args = {'raw': kw}
        # Generate options for each variant (cut or auw)
        for key in keys:
            kw = dict(options)
            if not kw.get(key, False):
                continue
            else:
                # Keep only the current variant, remove others
                for name in keys:
                    if name != key: kw.pop(name, None)  # keep only if spectrum is with cut (resp. auw)
                args[key] = kw
    else:
        # For non-spectrum stats, use single dictionary
        args = {'stat': options}
    return args


def _make_list_zrange(zranges):
    # Convert zrange to list of tuples
    if np.ndim(zranges[0]) == 0:
        zranges = [zranges]
    return list(zranges)


def compute_stats_from_options(stats, analysis='full_shape', cache=None,
                               get_stats_fn=tools.get_stats_fn,
                               get_catalog_fn=None,
                               read_clustering_catalog=tools.read_clustering_catalog,
                               read_full_catalog=tools.read_full_catalog,
                               **kwargs):
    """
    Compute summary statistics based on the provided options.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to compute.
        Choices: ['mesh2_spectrum', 'mesh3_spectrum', 'recon_mesh2_spectrum', 'window_mesh2_spectrum', 'window_mesh2_spectrum_fm', 'covariance_mesh2_spectrum']
    analysis : str, optional
        Type of analysis, 'full_shape' or 'png_local', to set fiducial options.
    cache : dict, optional
        Cache to store intermediate results (binning class and parent/reference random catalog).
        See :func:`spectrum2_tools.compute_mesh2_spectrum`, :func:`spectrum3_tools.compute_mesh3_spectrum`,
        and func:`tools.read_clustering_catalog` for details.
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    get_catalog_fn : callable, optional
        Function to get the filename for reading the catalog.
        If provided, it is given to ``read_clustering_catalog`` and ``read_full_catalog``.
    read_clustering_catalog : callable, optional
        Function to read the clustering catalog.
    read_full_catalog : callable, optional
        Function to read the full catalog.
    **kwargs : dict
        Options for catalog, reconstruction, and summary statistics.
    """
    # Ensure stats is a list (handle both string and list inputs)
    if isinstance(stats, str):
        stats = [stats]

    cache = cache or {}
    # Fill in fiducial defaults for all options
    options = fill_fiducial_options(kwargs, analysis=analysis)
    catalog_options = options['catalog']
    # tracers is a list of tracer1, tracer2, ... for cross-correlations
    tracers = list(catalog_options.keys())

    # Create redshift range lists for each tracer (support multiple z-bins)
    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}

    # Wrap catalog readers with catalog filename lookup function
    if get_catalog_fn is not None:
        read_clustering_catalog = functools.partial(read_clustering_catalog, get_catalog_fn=get_catalog_fn)
        read_full_catalog = functools.partial(read_full_catalog, get_catalog_fn=get_catalog_fn)

    # Check if any statistic requires reconstruction
    with_recon = any('recon' in stat for stat in stats)
    with_catalogs = True

    # Initialize catalogs and randoms dictionaries
    data, randoms = {}, {}
    with_stats_blinding = False
    if with_catalogs:
        # Load data and random catalogs for each tracer
        for tracer in tracers:
            _catalog_options = dict(catalog_options[tracer])
            # Expand redshift range to cover all requested z-bins
            _catalog_options['zrange'] = (min(zrange[0] for zrange in zranges[tracer]), max(zrange[1] for zrange in zranges[tracer]))

            # Add bitwise weight information (PIP, completeness) if needed
            if any(name in _catalog_options.get('weight', '') for name in ['bitwise', 'compntile']):
                # sets NTILE-MISSING-POWER (missing_power) and per-tile completeness (completeness)
                _catalog_options['binned_weight'] = read_full_catalog(kind='parent_data', **_catalog_options, attrs_only=True)

            # Add reconstruction options if needed
            if with_recon:
                recon_options = options['recon'][tracer]
                # pop as we don't need it anymore
                _catalog_options |= {key: recon_options.pop(key) for key in list(recon_options) if key in ['nran', 'zrange']}

            # Check if analysis requires blinding (e.g., protected samples)
            with_stats_blinding |= tools.check_if_stats_requires_blinding(analysis=analysis, **_catalog_options)
            # Prepare incomplete catalog handling if completeness weights provided
            if isinstance(_catalog_options.get('complete', None), dict):
                _catalog_options.setdefault('reshuffle', {})  # to pass on complete data

            # Read data and random catalogs
            data[tracer] = read_clustering_catalog(kind='data', **_catalog_options, concatenate=True)
            #_catalog_options.pop('complete', None)
            #_catalog_options.pop('reshuffle', None)
            randoms[tracer] = read_clustering_catalog(kind='randoms', **_catalog_options, cache=cache, concatenate=False)

    # Warn user if blinding will be applied
    if with_stats_blinding:
        warnings.warn('Output clustering statistics will be blinded on-the-fly.\nIf you do not want blinding, pass "protected" in the "analysis" argument.')

    # Initialize reconstruction attributes storage
    stat_recon_attrs = {}
    if with_recon:
        # data_rec, randoms_rec = {}, {}
        stat_recon_attrs = {'recon_mode': [], 'recon_smoothing_radius': []}
        for tracer in tracers:
            recon_options = dict(options['recon'][tracer])
            # Store reconstruction mode and radius for each tracer
            for name in stat_recon_attrs: stat_recon_attrs[name].append(recon_options[name[len('recon_'):]])

            # Run reconstruction to get shifted positions
            data[tracer]['POSITION_REC'], randoms_rec_positions = compute_reconstruction(lambda: {'data': data[tracer], 'randoms': Catalog.concatenate(randoms[tracer])}, **recon_options)

            # Assign reconstructed positions to random catalogs
            start = 0
            for random in randoms[tracer]:
                size = len(random['POSITION'])
                random['POSITION_REC'] = randoms_rec_positions[start:start + size]
                start += size
            # Keep only the requested number of random files (for reduced memory footprint)
            randoms[tracer] = randoms[tracer][:catalog_options[tracer]['nran']]  # keep only relevant random files

    # Compute angular upweights for fiber collision corrections if requested
    if with_catalogs and any(options[stat].get('auw', False) for stat in stats):

        def get_data(tracer):
            # Load full parent catalogs (before any selection) for AUW computation
            _catalog_options = catalog_options[tracer] | dict(zrange=None)
            return {kind: read_full_catalog(kind=kind, **_catalog_options) for kind in ['fibered_data', 'parent_data']}

        # Compute angular upweights from fibered vs parent catalogs
        auw = compute_angular_upweights(*[functools.partial(get_data, tracer) for tracer in tracers])
        fn_catalog_options = {tracer: catalog_options[tracer] | dict(zrange=None) for tracer in tracers}
        fn = get_stats_fn(kind='particle2_angular_upweights', catalog=fn_catalog_options)
        # Write computed angular upweights to disk
        tools.write_stats(fn, auw)
        # Update all statistics options with computed angular upweights
        for key, kw in options.items():
            if kw.get('auw', False): kw['auw'] = auw  # update with angular upweights

    # Loop over all requested redshift bins
    for zvals in zip(*(zranges[tracer] for tracer in tracers)):
        zrange = dict(zip(tracers, zvals))

        def get_zcatalog(catalog, zrange):
            # Extract redshift slice from catalog
            mask = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])
            return catalog[mask]

        zdata, zrandoms = {}, {}
        if with_catalogs:
            # Slice catalogs to current redshift bin
            for tracer in tracers:
                zdata[tracer] = get_zcatalog(data[tracer], zrange[tracer])
                zrandoms[tracer] = [get_zcatalog(random, zrange[tracer]) for random in randoms[tracer]]
        fn_catalog_options = {tracer: catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}

        def get_catalog_recon(catalog):
            # Replace positions with reconstructed positions
            return catalog.clone(POSITION=catalog['POSITION_REC'])

        # Summary statistics computation loop
        for recon in ['', 'recon_']:
            stat = f'{recon}particle2_correlation'
            if stat in stats:
                correlation_options = dict(options[stat])

                def get_data(tracer):
                    # Prepare data structure for correlation function measurement
                    if recon:
                        # Use reconstructed positions as primary, randoms for random catalogs
                        return {'data': get_catalog_recon(zdata[tracer]),
                                'randoms': zrandoms[tracer],
                                'shifted': [get_catalog_recon(zrandom) for zrandom in zrandoms[tracer]]}
                    # Default: use original positions
                    return {'data': zdata[tracer], 'randoms': zrandoms[tracer]}

                # Compute 2-point correlation function
                correlation = compute_particle2_correlation(*[functools.partial(get_data, tracer) for tracer in tracers], **correlation_options)
                # Apply blinding if requested (only for raw measurements, not reconstruction)
                if with_stats_blinding and not recon:  # FIXME
                    correlation = tools.apply_blinding(correlation, tracers, zrange=sum(zrange.values(), start=tuple()))
                # Store reconstruction metadata
                if recon:
                    correlation.attrs.update(stat_recon_attrs)
                # Write correlation to disk
                fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **correlation_options)
                tools.write_stats(fn, correlation)

            # Map of spectrum statistics to computation functions
            funcs = {f'{recon}mesh2_spectrum': compute_mesh2_spectrum, f'{recon}mesh3_spectrum': compute_mesh3_spectrum}

            for stat, func in funcs.items():
                if stat in stats:
                    spectrum_options = dict(options[stat])
                    # Extract selection weights if provided (e.g., angular selection weights)
                    selection_weights = spectrum_options.pop('selection_weights', None)

                    def get_data(tracer):
                        # Prepare data for spectrum measurement
                        # Concatenate all random catalogs into single object
                        czrandoms = Catalog.concatenate(zrandoms[tracer])
                        if recon:
                            # Use reconstructed positions, with same shifts applied to randoms
                            toret = {'data': get_catalog_recon(zdata[tracer]),
                                     'randoms': czrandoms,
                                     'shifted': get_catalog_recon(czrandoms)}
                        else:
                            # Default: use original positions
                            toret = {'data': zdata[tracer], 'randoms': czrandoms}
                        # Apply selection weights if provided (for bispectrum, NZ**(1. / 3.) weighting, etc.)
                        if selection_weights:
                            toret = {name: selection_weights[tracer](catalog) for name, catalog in toret.items()}
                        return toret

                    # Compute power spectrum or bispectrum
                    spectrum = func(*[functools.partial(get_data, tracer) for tracer in tracers], cache=cache, **spectrum_options)
                    # Ensure spectrum is a dictionary (may contain raw, cut, auw variants)
                    if not isinstance(spectrum, dict): spectrum = {'raw': spectrum}

                    # Write all spectrum variants to disk
                    for key, kw in _expand_cut_auw_options(stat, spectrum_options).items():
                        fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                        # Apply blinding if requested
                        if with_stats_blinding:
                            spectrum[key] = tools.apply_blinding(spectrum[key], tracers, zrange=sum(zrange.values(), start=tuple()))
                        # Store reconstruction metadata
                        if recon:
                            spectrum.attrs.update(stat_recon_attrs)
                        tools.write_stats(fn, spectrum[key])

        # Synchronize across all processes before proceeding to windows
        jax.experimental.multihost_utils.sync_global_devices('spectrum')  # wait for the writer

        # Window matrix
        funcs = {"window_mesh2_spectrum": compute_window_mesh2_spectrum,
                 "window_mesh3_spectrum": compute_window_mesh3_spectrum}

        for stat, func in funcs.items():
            if stat in stats:
                window_options = dict(options[stat])
                # Extract selection weights if provided
                selection_weights = window_options.pop('selection_weights', None)

                def get_data(tracer):
                    # Prepare randoms for window function computation
                    czrandoms = Catalog.concatenate(zrandoms[tracer])
                    toret = {'data': zdata[tracer], 'randoms': czrandoms}
                    # Apply selection weights if provided
                    if selection_weights:
                        toret = {name: selection_weights[tracer](catalog) for name, catalog in toret.items()}
                    return toret

                # Load measured spectrum (or compute if not provided)
                spectrum_fn = window_options.pop('spectrum', None)
                fn_window_options = window_options | dict(auw=False, cut=False)
                if spectrum_fn is None:
                    # Auto-detect spectrum filename from options
                    spectrum_stat = stat.replace("window_", "")
                    fn_window_options = options[spectrum_stat] | fn_window_options
                    spectrum_fn = get_stats_fn(kind=spectrum_stat, catalog=fn_catalog_options, **(options[spectrum_stat] | dict(auw=False, cut=False)))
                spectrum = types.read(spectrum_fn)

                def get_extra(ibatch, nbatch):
                    # Generate batch identifier string for window correlation functions
                    return f'batch-{ibatch:d}-{nbatch:d}'

                # Check if computing window in batches (for memory efficiency)
                ibatch = window_options.get('ibatch', None)
                extra = get_extra(*ibatch) if ibatch is not None else None

                # Load previously computed batch windows if continuing
                nbatch = window_options.get('computed_batches', False)
                if nbatch:
                    # Load window multipole batches computed in previous runs
                    fns = [get_stats_fn(kind=f'{stat.replace("_spectrum", "")}_correlation_raw', catalog=fn_catalog_options, **(fn_window_options | dict(extra=get_extra(ibatch, nbatch)))) for ibatch in range(nbatch)]
                    window_options['computed_batches'] = [types.read(fn) for fn in fns]
                # Remove basis from options (will be extracted from spectrum)
                window_options.pop('basis', None)

                # Compute window function
                window = func(*[functools.partial(get_data, tracer) for tracer in tracers], spectrum=spectrum, **window_options)

                # Write window matrix to disk
                for key, kw in _expand_cut_auw_options(stat, fn_window_options).items():
                    if key in window:
                        # Extract basis from spectrum if available
                        basis = getattr(next(iter(window[key].observable)), 'basis', None)
                        if basis is not None: kw['basis'] = basis
                        fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                        tools.write_stats(fn, window[key])

                # Write raw correlation functions (intermediate products) to disk
                for key in window:
                    if 'correlation' in key:  # window functions
                        fn = get_stats_fn(kind=key, catalog=fn_catalog_options, **(fn_window_options | dict(extra=extra)))
                        tools.write_stats(fn, window[key])

        # Synchronize before window forward model computation
        jax.experimental.multihost_utils.sync_global_devices('window')  # wait for the writer

        # Window matrix using forward model (for RIC and AMR effects)
        funcs = {"window_mesh2_spectrum_fm": compute_window_mesh2_spectrum_fm}
        for stat, func in funcs.items():
            if stat in stats:
                # Forward model window only supports auto-correlations currently
                if len(tracers) > 1:
                    raise NotImplementedError("Forward model window function not yet implemented for cross-correlations")

                window_options = dict(options[stat])
                selection_weights = window_options.pop("selection_weights", None)

                def get_data(tracer):
                    # Prepare randoms for forward model window computation
                    czrandoms = Catalog.concatenate(zrandoms[tracer])
                    toret = {"data": zdata[tracer], "randoms": czrandoms}
                    if selection_weights:
                        toret = {name: selection_weights[tracer](catalog) for name, catalog in toret.items()}
                    return toret

                def _check_fn(fn, tracers, name=""):
                    # Convert single filename to dictionary for tracer indexing
                    if len(tracers) == 1:
                        fn = {(tracer, tracer): fn for tracer in tracers}
                    else:
                        raise ValueError(f"provide a dictionary of (tracer1, tracer2): {name} for tracer1, tracer2 in {tracers}")
                    return fn

                def _read_tracer(fns, tracers2):
                    # Read spectrum/window for specific tracer pair (handle ordering)
                    if tracers2 not in fns:
                        tracers2 = tracers2[::-1]
                    return types.read(fns[tracers2])

                # Get fiducial theory for computing forward model derivatives
                theory_stat = stat.replace("window_", "theory_").replace("_fm", "")
                theory_fn = window_options.pop("theory", None)

                if theory_fn is None:
                    # Auto-compute fiducial theory from spectrum and window
                    products_fn = {spectrum_region: {} for spectrum_region in window_options["spectrum_regions"]}
                    # Collect power spectrum and window, for each region if relevant
                    for spectrum_region in window_options["spectrum_regions"]:
                        for name in ["spectrum", "window"]:
                            kind_stat = (
                                stat.replace("window_", "").replace("_fm", "") if name == "spectrum" else stat.replace("window_", f"{name}_").replace("_fm", "")
                            )
                            fn = window_options.pop(name, None)
                            if fn is None:
                                # Auto-detect measurement filename
                                kw = options[kind_stat] | {"auw": False, "cut": False}
                                fn = get_stats_fn(
                                    kind=kind_stat,
                                    catalog=fn_catalog_options[tracers[0]],
                                    **kw | {"region": spectrum_region},
                                )
                            products_fn[spectrum_region][name] = fn

                    # Load spectra and windows from disk
                    spectra = [types.read(products_fn[spectrum_region]["spectrum"]) for spectrum_region in window_options["spectrum_regions"]]
                    windows = [types.read(products_fn[spectrum_region]["window"]) for spectrum_region in window_options["spectrum_regions"]]
                    # Combine measurements from multiple regions and fit for theory
                    theory = types.sum(
                        [run_preliminary_fit_mesh2_spectrum(data=spectrum, window=window) for spectrum, window in zip(spectra, windows, strict=True)]
                    )
                    spectrum = types.sum(spectra)
                    theory_fn = get_stats_fn(kind=theory_stat, catalog=(fn_catalog_options[tracers[0]]))
                    tools.write_stats(theory_fn, theory)

                # Synchronize before reading theory
                jax.experimental.multihost_utils.sync_global_devices("theory")  # such that theory ready for window
                theory = types.read(theory_fn)

                # Load example of output measurement. If spectrum_fn provided, use it; otherwise use spectrum loaded for the preliminary fit in the theory block above
                spectrum_fn = window_options.pop("spectrum", None)
                fn_window_options = window_options | {"auw": False, "cut": False}
                if spectrum_fn is None:
                    spectrum_fn = {}
                    spectrum_stat = stat.replace("window_", "").replace("_fm", "")
                    for spectrum_region in window_options["spectrum_regions"]:
                        fn_window_options = options[spectrum_stat] | fn_window_options
                        spectrum_fn[spectrum_region] = get_stats_fn(
                            kind=spectrum_stat, catalog=fn_catalog_options,
                            **(options[spectrum_stat] | {"auw": False, "cut": False} | {"region": spectrum_region}))
                    spectrum = types.sum([types.read(spectrum_fn[spectrum_region]) for spectrum_region in window_options["spectrum_regions"]])
                else:
                    spectrum = types.read(spectrum_fn)

                # Now compute window function using forward model with derivatives
                window = func(*[functools.partial(get_data, tracer) for tracer in tracers], spectrum=spectrum, theory=theory, **window_options)
                # This is a dict of dict of lists of windows : {modeled_effect:{spectrum_region:[window, ...], ...}, ...}
                for effect in window:  # geo, RIC or RIC+AMR
                    for spectrum_region in window[effect]:  # eg NGC, SGC
                        for i, seed in enumerate(window_options["seeds"]):
                            if window_options['ellsout'] is None:
                                extra = f"{effect}_seed={seed}"
                            else:
                                listell = "".join(map(str, window_options['ellsout']))
                                extra = f'{effect}_{listell}_seed={seed}'

                            options = fn_window_options | {"extra": extra, "region": spectrum_region}
                            tools.write_stats(get_stats_fn(kind=stat, catalog=fn_catalog_options, **options), window[effect][spectrum_region][i])

        # Covariance matrix computation
        funcs = {'covariance_mesh2_spectrum': compute_covariance_mesh2_spectrum}
        for stat, func in funcs.items():
            if stat in stats:
                covariance_options = dict(options[stat])
                theory_stat = stat.replace('covariance_', 'theory_')
                theory_fn = covariance_options.pop('theory', None)

                def get_data(tracer):
                    # Prepare catalogs for covariance computation
                    czrandoms = Catalog.concatenate(zrandoms[tracer])
                    return {'data': zdata[tracer], 'randoms': czrandoms}

                def _check_fn(fn, tracers, name=''):
                    # Convert single filename to tracer pair dictionary
                    if len(tracers) == 1:
                        fn = {(tracer, tracer): fn for tracer in tracers}
                    else:
                        raise ValueError(f'provide a dictionary of (tracer1, tracer2): {name} for tracer1, tracer2 in {tracers}')
                    return fn

                def _read_tracer(fns, tracers2):
                    # Read file for tracer pair (handle ordering)
                    if tracers2 not in fns: tracers2 = tracers2[::-1]
                    return types.read(fns[tracers2])

                if theory_fn is None:
                    # Auto-compute fiducial theory from spectrum and window
                    products_fn = {}
                    # Collect power spectrum and window
                    for name in ['spectrum', 'window']:
                        kind_stat = stat.replace('covariance_', '') if name == 'spectrum' else stat.replace('covariance_', f'{name}_')
                        fn = covariance_options.pop(name, None)
                        if fn is None:
                            # Auto-detect measurement files for each tracer pair
                            kw = options[kind_stat] | dict(auw=False, cut=False)
                            fn = {(tracer, tracer): get_stats_fn(kind=kind_stat, catalog=fn_catalog_options[tracer], **kw) for tracer in tracers}
                            # Add cross-correlation file if multiple tracers
                            if len(tracers) > 1:
                                fn[tuple(tracers)] = get_stats_fn(kind=kind_stat, catalog=fn_catalog_options, **kw)
                        elif not isinstance(fn, dict):
                            _check_fn(fn, tracers, name=name)
                        products_fn[name] = fn

                    # Compute theory for each tracer pair
                    theory_fn = {}
                    for tracers2 in itertools.combinations_with_replacement(tracers, r=2):
                        spectrum = _read_tracer(products_fn['spectrum'], tracers2)
                        window = _read_tracer(products_fn['window'], tracers2)
                        # Fit theory to measurement (preliminary fit for covariance)
                        theory = run_preliminary_fit_mesh2_spectrum(data=spectrum, window=window)
                        theory_fn[tracers2] = get_stats_fn(kind=theory_stat, catalog=(fn_catalog_options[tracers2[0]] if tracers2[1] == tracers2[0] else {tracer: fn_catalog_options[tracer] for tracer in tracers2}))
                        # Write theory to disk
                        tools.write_stats(theory_fn[tracers2], theory)
                else:
                    _check_fn(theory_fn, tracers, name='theory')

                # Synchronize before reading theory
                jax.experimental.multihost_utils.sync_global_devices('theory')  # such that theory ready for window

                # Load theory for all tracer pairs
                fields = {tracer: tools.get_simple_tracer(tracer) for tracer in tracers}
                theory = {tuple(fields[tracer] for tracer in tracers2): _read_tracer(theory_fn, tracers2) for tracers2 in itertools.product(tracers, repeat=2)}
                theory = types.ObservableTree(list(theory.values()), fields=list(theory.keys()))

                # Compute covariance matrix
                covariance = func(*[functools.partial(get_data, tracer) for tracer in tracers], theory=theory, fields=list(fields.values()), **covariance_options)

                # Write covariance matrix to disk
                for key, kw in _expand_cut_auw_options(stat, covariance_options).items():
                    fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                    if key in covariance:
                        tools.write_stats(fn, covariance[key])

                # Write intermediate correlation functions to disk
                for key in covariance:
                    if 'correlation' in key:  # window functions
                        fn = get_stats_fn(kind=key, catalog=fn_catalog_options, **(covariance_options | dict(auw=False, cut=False)))
                        tools.write_stats(fn, covariance[key])



def list_stats(stats, get_stats_fn=tools.get_stats_fn, **kwargs):
    """
    List measurements produced by :func:`compute_stats_from_options`.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to list.
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for catalog and summary statistics. For example:
            catalog = dict(version='holi-v1-altmtl', tracer='LRG', zrange=[(0.4, 0.6), (0.8, 1.1)], imock=451)
            mesh2_spectrum = dict(cut=True, auw=True, ells=(0, 2, 4), mattrs=dict(boxsize=7000., cellsize=8.))  # all arguments for compute_mesh2_spectrum
            mesh3_spectrum = dict(basis='sugiyama-diagonal', ells=[(0, 0, 0)], mattrs=dict(boxsize=7000., cellsize=10.))  # all arguments for compute_mesh3_spectrum
    """
    # Ensure stats is a list
    if isinstance(stats, str):
        stats = [stats]

    # Fill fiducial defaults
    kwargs = fill_fiducial_options(kwargs)
    catalog_options = kwargs['catalog']
    tracers = list(catalog_options.keys())
    # Build list of redshift ranges for each tracer
    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}

    toret = {stat: [] for stat in stats}
    # Iterate over all combinations of redshift bins and statistics
    for zvals in zip(*(zranges[tracer] for tracer in tracers)):
        zrange = dict(zip(tracers, zvals))
        _catalog_options = {tracer: catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}
        for stat in stats:
            # Generate option combinations (raw, cut, auw)
            for kw in _expand_cut_auw_options(stat, kwargs[stat]).values():
                kw = dict(catalog=_catalog_options, **kw)
                fn = get_stats_fn(kind=stat, **kw)
                toret[stat].append((fn, kw))
    return toret


def postprocess_stats_from_options(postprocess, analysis='full_shape', get_stats_fn=tools.get_stats_fn, **kwargs):
    """
    Postprocess summary statistics based on the provided options.

    Parameters
    ----------
    postprocess : str or list of str
        Postprocessing.
        Choices: ['combine_regions', 'rotation_mesh2_spectrum']
    analysis : str, optional
        Type of analysis, 'full_shape' or 'png_local', to set fiducial options.
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for summary statistics, and choices in ``postprocess``.
    """
    # Ensure postprocess is a list
    if isinstance(postprocess, str):
        postprocess = [postprocess]

    imocks = kwargs.pop('imocks', None)
    # Fill fiducial defaults
    options = fill_fiducial_options(kwargs, analysis=analysis)
    catalog_options = options['catalog']
    tracers = list(catalog_options.keys())
    # Set default region to combined
    for tracer in tracers:
        catalog_options[tracer].setdefault('region', 'GCcomb')  # default, for rotation, rotate
    # Build redshift range lists
    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}
    # Default imock if not specified
    if imocks is None: imocks = [catalog_options[tracers[0]].get('imock', None)]

    def _iter_on_mocks(options):
        # Helper to iterate over multiple mock realizations
        _options = copy.deepcopy(options)
        for imock in imocks:
            for tracer in _options['catalog']:
                _options['catalog'][tracer]['imock'] = imock
            yield _options

    # Loop over redshift bins
    for zvals in zip(*(zranges[tracer] for tracer in tracers)):
        zrange = dict(zip(tracers, zvals))
        fn_catalog_options = {tracer: catalog_options[tracer] | dict(zrange=zrange[tracer]) for tracer in tracers}

        if 'combine_regions' in postprocess:
            # Combine measurements from different sky regions (NGC, SGC)
            combine_options = dict(options.get('combine_regions', {}))
            regions = combine_options.pop('regions', ['NGC', 'SGC'])
            stats = combine_options.pop('stats', ['mesh2_spectrum', 'mesh3_spectrum'])

            def _combine_stats(stat, region_comb, regions, get_stats_fn=get_stats_fn, **options):
                # Helper to combine statistics from multiple regions
                all_fns = {}
                # List all measurement files for each region
                for region in regions + [region_comb]:
                    kwargs = dict(options)
                    kwargs['catalog'] = {tracer: options['catalog'][tracer] | dict(region=region) for tracer in options['catalog']}
                    all_fns[region] = list_stats(stat, get_stats_fn=get_stats_fn, **kwargs)
                stats = next(iter(all_fns.values())).keys()
                # Combine each statistic variant
                for stat in stats:
                    for ifn, (fn_comb, _) in enumerate(all_fns[region_comb][stat]):
                        fns = [all_fns[region][stat][ifn][0] for region in regions]  # [1] is kwargs
                        exists = {os.path.exists(fn): fn for fn in fns}
                        if all(exists):
                            # Read and combine measurements from different regions
                            combined = tools.combine_stats([types.read(fn) for fn in fns])
                            tools.write_stats(fn_comb, combined)
                        else:
                            logger.info(f'Skipping {fn_comb} as {[fn for ex, fn in exists.items() if not ex]} do not exist')

            # Get all possible region combinations
            for region_comb, regions in tools.possible_combine_regions(regions).items():
                for stat in stats:
                    if 'window' in stat or 'covariance' in stat:
                        # Window and covariance don't need to loop over mocks
                        _combine_stats(stat, region_comb, regions, get_stats_fn=get_stats_fn, **options)
                    else:
                        # Measurements need to loop over mocks
                        for _options in _iter_on_mocks(options | dict(catalog=fn_catalog_options)):
                            _combine_stats(stat, region_comb, regions, get_stats_fn=get_stats_fn, **_options)

        if 'rotation_mesh2_spectrum' in postprocess:
            # Compute rotation matrix for power spectrum (corrections for systematic effects)
            stat = 'rotation_mesh2_spectrum'
            kind_stat = stat.replace('rotation_', '')
            rotation_options = dict(options.get(stat, {}))
            products = {}
            # Read window and covariance (required for rotation computation)
            for name, kind in zip(['window', 'covariance'], [f'window_{kind_stat}', f'covariance_{kind_stat}']):
                fn = rotation_options.pop(name, None)
                if fn is None:
                    # Auto-detect window/covariance filenames
                    # FIXME, in case covariance with theta-cut is available
                    kw = options.get(kind_stat, {}) | dict(auw=False, cut=(name == 'window'))
                    fn = get_stats_fn(kind=kind, catalog=fn_catalog_options, **kw)
                # Read from disk or use provided object
                if isinstance(fn, types.ObservableTree):
                    products[name] = fn
                else:
                    products[name] = types.read(fn)

            # Select auto-covariance for single tracer
            tracers2 = tuple(tracers * (2 // len(tracers)))
            #print(products['covariance'].observable.labels())
            products['covariance'] = products['covariance'].at.observable.get(observables=tools.get_simple_stats(kind_stat), tracers=tuple(tools.get_simple_tracer(tracer) for tracer in tracers2))

            # Read or compute data and theory measurements
            for name in ['data', 'theory']:
                fns = rotation_options.pop(name, {})
                if isinstance(fns, dict):
                    # Auto-detect filenames
                    kw = dict(catalog=fn_catalog_options) | options.get(kind_stat, {}) | dict(auw=False, cut=(name == 'data')) | fns
                    fns = get_stats_fn(kind=kind_stat, **kw)
                # Read from disk or use provided object
                if isinstance(fns, types.ObservableTree):
                    products[name] = fns
                else:
                    # Read and average multiple measurements
                    if isinstance(fns, (str, Path)): fns = [fns]
                    products[name] = types.mean([types.read(fn) for fn in fns])

            # Compute rotation on single process only (not parallelized)
            if jax.process_index() == 0:
                rotation = compute_rotation_mesh2_spectrum(**products, **rotation_options)
                # Save rotation matrix to disk
                kw = options.get(kind_stat, {}) | dict(auw=False, cut=True)
                fn = get_stats_fn(kind=stat, catalog=fn_catalog_options, **kw)
                tools.write_stats(fn, rotation)


def combine_stats_from_options(stats, region_comb, regions, get_stats_fn=tools.get_stats_fn, **kwargs):
    """
    Combine summary statistics from multiple regions based on the provided options.

    Warning
    --------
    Use postprocess_from_options(['combine_regions']) instead.

    Parameters
    ----------
    stats : str or list of str
        Summary statistics to combine.
    region_comb : str
        Combined region name, e.g. 'GCcomb'.
    regions : list of str
        Regions to combine, e.g. ['NGC', 'SGC'].
    get_stats_fn : callable, optional
        Function to get the filename for storing the measurement.
    **kwargs : dict
        Options for catalog and summary statistics. For example:
            catalog = dict(version='holi-v1-altmtl', tracer='LRG', zrange=[(0.4, 0.6), (0.8, 1.1)], imock=451)
            mesh2_spectrum = dict(cut=True, auw=True, ells=(0, 2, 4), mattrs=dict(boxsize=7000., cellsize=8.))  # all arguments for compute_mesh2_spectrum
            mesh3_spectrum = dict(basis='sugiyama-diagonal', ells=[(0, 0, 0)], mattrs=dict(boxsize=7000., cellsize=10.))  # all arguments for compute_mesh3_spectrum
    """
    warnings.warn("deprecated; use postprocess_from_options(['combine_regions']) instead")
    options = fill_fiducial_options(kwargs)
    regions = list(regions)
    all_fns = {}
    # List all measurement files for each region
    for region in regions + [region_comb]:
        kwargs = dict(options)
        kwargs['catalog'] = {tracer: options['catalog'][tracer] | dict(region=region) for tracer in options['catalog']}
        all_fns[region] = list_stats(stats, get_stats_fn=get_stats_fn, **kwargs)

    stats = next(iter(all_fns.values())).keys()
    # Combine each statistic
    for stat in stats:
        for ifn, (fn_comb, _) in enumerate(all_fns[region_comb][stat]):
            fns = [all_fns[region][stat][ifn][0] for region in regions]  # [1] is kwargs
            exists = {os.path.exists(fn): fn for fn in fns}
            if all(exists):
                # Read and combine measurements from different regions
                combined = tools.combine_stats([types.read(fn) for fn in fns])
                tools.write_stats(fn_comb, combined)
            else:
                logger.debug(f'Skipping {fn_comb} as {[fn for ex, fn in exists.items() if not ex]} do not exist')


def main(**kwargs):
    r"""
    This is an example main, which can be run from command line to compute fiducial statistics.
    Let's try to keep it simple; write your own if you need anything fancier.
    Or just use :func:`compute_stats_from_options` directly; see example in :mod:`job_scripts/desipipe_holi_mocks.py`.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', help='what do you want to compute?', type=str, nargs='*', choices=['mesh2_spectrum', 'mesh3_spectrum', 'particle2_correlation', 'recon_particle2_correlation', 'window_mesh2_spectrum', 'window_mesh3_spectrum'], default=['mesh2_spectrum'])
    parser.add_argument('--version', help='catalog version; e.g. holi-v1-altmtl', type=str, default=None)
    parser.add_argument('--cat_dir', help='where to find catalogs', type=str, default=None)
    parser.add_argument('--tracer', help='tracer(s) to be selected - e.g. LRG ELG for cross-correlation', nargs='*', type=str, default='LRG')
    parser.add_argument('--zrange', help='redshift bins; 0.4 0.6 0.8 1.1 to run (0.4, 0.6), (0.8, 1.1)', nargs='*', type=float, default=None)
    parser.add_argument('--imock', help='mock number', type=int, nargs='*', default=[None])
    parser.add_argument('--region', help='regions', type=str, nargs='*', choices=['N', 'S', 'NGC', 'SGC', 'NGCnoN', 'SGCnoDES'], default=['NGC', 'SGC'])
    parser.add_argument('--analysis', help='type of analysis', type=str, choices=['full_shape', 'png_local', 'full_shape_protected'], default='full_shape')
    parser.add_argument('--weight',  help='type of weights to use for tracer; "default" just uses WEIGHT column', type=str, default='default-FKP')
    parser.add_argument('--thetacut',  help='Apply theta-cut', action='store_true', default=None)
    parser.add_argument('--auw',  help='Apply angular upweighting', action='store_true', default=None)
    parser.add_argument('--boxsize',  help='box size', type=float, default=None)
    parser.add_argument('--cellsize', help='cell size', type=float, default=None)
    parser.add_argument('--nran', help='number of random files to combine together (1-18 available)', type=int, default=None)
    parser.add_argument('--make_complete', help='make on-the-fly (completeness-weighted) complete catalogs', action='store_true', default=None)
    parser.add_argument('--expand_randoms', help='expand catalog of randoms; provide version of parent randoms (must be registered in get_catalog_fn)', type=str, default=None)
    meas_dir = Path(os.getenv('SCRATCH')) / 'measurements'
    parser.add_argument('--stats_dir',  help=f'base directory for measurements, default is {meas_dir}', type=str, default=meas_dir)
    parser.add_argument('--stats_extra',  help='extra string to include in measurement filename', type=str, default='')
    parser.add_argument('--combine', help='combine measurements in two regions', type=str, nargs='*', default=None, choices=['mesh2_spectrum', 'mesh3_spectrum', 'particle2_correlation', 'recon_particle2_correlation', 'window_mesh2_spectrum', 'window_mesh3_spectrum'])

    args = parser.parse_args()
    # Set JAX to use 90% of GPU memory (leave 10% for overhead)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    import jax
    # Initialize distributed JAX if computing statistics
    if args.stats:
        jax.distributed.initialize()

    # Set up logging
    setup_logging()

    # Get default redshift ranges for tracer and analysis type
    if args.zrange is None:
        zranges = tools.propose_fiducial('zranges', tracer=tools.join_tracers(args.tracer), analysis=args.analysis)
    else:
        # Parse redshift range from command line (pairs of values)
        assert len(args.zrange) % 2 == 0
        zranges = list(zip(args.zrange[::2], args.zrange[1::2]))

    # Build mesh options (boxsize and cellsize)
    mattrs = {key: value for key, value in dict(boxsize=args.boxsize, cellsize=args.cellsize).items() if value is not None}
    options = {'mattrs': mattrs}
    # Apply theta-cut and angular upweighting options if requested
    for stat in ['mesh2_spectrum', 'particle2_correlation']:
        options.setdefault(stat, {})
        options[stat].update(cut=args.thetacut, auw=args.auw)

    # Set up catalog filename lookup function
    get_catalog_fn = tools.get_catalog_fn
    # Set up statistics filename generation function with custom directory and extra string
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=args.stats_dir, extra=args.stats_extra)
    cache = {}

    def _keep_if_not_none(**kwargs):
        # Helper to filter out None values from kwargs
        return {k: v for k, v in kwargs.items() if v is not None}

    # Build catalog options from command-line arguments
    catalog_options = dict(tracer=args.tracer, zrange=zranges, ext=None)
    catalog_options |= _keep_if_not_none(weight=args.weight, version=args.version, cat_dir=args.cat_dir, nran=args.nran)
    # Merge all options and fill fiducial defaults
    options = _merge_options(fill_fiducial_options(dict(catalog=catalog_options) | options, analysis=args.analysis), kwargs)

    # Compute statistics for each mock realization and sky region
    if args.stats:
        for imock in args.imock:
            for region in args.region:
                _options_imock = dict(options)
                for tracer in _options_imock['catalog']:
                    _options_imock['catalog'][tracer] = _options_imock['catalog'][tracer] | dict(region=region, imock=imock)
                    # Add expanded random catalog if requested (not all columns saved)
                    if args.expand_randoms:
                        _options_imock['catalog'][tracer]['expand'] = {'parent_randoms_fn': get_catalog_fn(kind='parent_randoms', version=args.expand_randoms, tracer=tracer, region=region, nran=max(value['nran'] for value in _options_imock['recon'].values()))}
                    # Enable completeness-weighted catalogs if requested
                    if args.make_complete:
                        _options_imock['catalog'][tracer]['complete'] = {}
                # Compute all requested statistics
                compute_stats_from_options(args.stats, get_catalog_fn=get_catalog_fn, get_stats_fn=get_stats_fn, cache=cache, analysis=args.analysis, **_options_imock)
                # Synchronize all processes before next region
                jax.experimental.multihost_utils.sync_global_devices(region)

    # Postprocess statistics (combine regions, compute rotations)
    if args.combine is not None and jax.process_index() == 0:
        stats = []
        if args.combine: stats = args.combine
        elif args.stats: stats = args.stats
        else: stats = ['mesh2_spectrum', 'mesh3_spectrum']  # best guess, if not argument was provided
        postprocess_stats_from_options(['combine_regions'], get_stats_fn=get_stats_fn, combine_regions=dict(stats=stats), **options, imocks=args.imock)

    # Shutdown distributed JAX
    if args.stats:
        jax.distributed.shutdown()


if __name__ == '__main__':

    from jax import config
    # Enable 64-bit precision for higher accuracy (slower computation)
    config.update('jax_enable_x64', False)
    main()