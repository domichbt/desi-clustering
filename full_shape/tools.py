"""
In this file we propose a dictionary-like interface to building :mod:`desilike` likelihoods.

Important functions are:
- :func:`generate_likelihood_options_helper`, helper to generate dictionary of options
- :func:`get_stats`: read clustering statistics from disk
- :func:`get_theory`: build desilike theory
- :func:`get_single_likelihood` return single desilike likelihood (that can be summed up with others)
- :func:`get_fits_fn`: proposed path to output fits.

Dictionary of options are organized as follows:
- list of dictionaries, one for each independent (summed) likelihood;
- each of this dictionary is {'observables': [observable1, observable2, ...], 'covariance': covariance options}
- each of observable1, observable2, ... is a dictionary that specifies how to build the desilike
observable (data, theory, window): ``{'stat': {'kind': ..., 'basis': ..., 'select': [...]},
'catalog': {'version':, ...}, 'theory': {'model': ...}, 'window': {}}``.
"""


import os
import re
import json
import hashlib
import logging
import numbers
import itertools
import warnings
from pathlib import Path

import numpy as np
import scipy as sp
import lsstypes as types
from lsstypes.utils import get_hartlap2007_factor, get_percival2014_factor, mkdir

from clustering_statistics.tools import (write_stats, float2str, get_full_tracer, get_simple_tracer, _make_tuple,
                                         get_simple_stats, _unzip_catalog_options, default_mpicomm, setup_logging)
from clustering_statistics import tools as clustering_tools


logger = logging.getLogger('tools')


_fiducial = None

def get_fiducial():
    global _fiducial
    if _fiducial is None:
        from cosmoprimo.fiducial import DESI
        _fiducial = DESI()
    return _fiducial


def get_cosmology(cosmology_options: dict=None):
    """
    Construct and return a :mod:`desilike` :class:`Cosmoprimo` calculator.

    Returns
    -------
    cosmo : :class:`desilike.theories.Cosmoprimo`
        Instance with configured priors.
    """
    from desilike.theories import Cosmoprimo
    if isinstance(cosmology_options, Cosmoprimo):
        return cosmology_options
    cosmology_options = cosmology_options or {}
    model = cosmology_options.get('model', 'base_ns-fixed')
    cosmo = Cosmoprimo(engine='class', fiducial=get_fiducial())
    # Free parameters h, omega_cdm, omega_b, logA with uniform priors
    # n_s and tau_reio are fixed
    # A Gaussian prior on omega_b.
    params = {
        'H0':       {'derived': True},
        'Omega_m':  {'derived': True},
        'sigma8_m': {'derived': True},
        'tau_reio': {'fixed': True},
        'n_s':      {'fixed': 'ns-fixed' in model},
        'omega_b':  {'fixed': False, 'prior': {'dist': 'norm', 'loc': 0.02237,  'scale': 0.00037}},
        'h':        {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.5,  0.9]}},
        'omega_cdm':{'fixed': False, 'prior': {'dist': 'uniform', 'limits': [0.05, 0.2]}},
        'logA':     {'fixed': False, 'prior': {'dist': 'uniform', 'limits': [2.0,  4.0]}},
    }
    if 'w0wa' in model:
        params['w0_fld'] = {'fixed': False}
        params['wa_fld'] = {'fixed': False}
    for name, config in params.items():
        if name in cosmo.init.params:
            cosmo.init.params[name].update(**config)
        else:
            cosmo.init.params[name] = config
    return cosmo



def _get_default_theory_nuisance_priors(model, stat, prior_basis, b3_coev=True, tracer=None, sigma8_fid=1.):
    """
    Build a dictionary of parameter priors.

    Parameters
    ----------
    model : str
        Perturbation theory model tag. When 'EFT', FoG parameters are fixed.
    stat : str
        Observable; one of ['mesh2_spectrum', 'mesh2_spectrum'].
    prior_basis : str
        'physical' or 'physical_aap' uses physical bias parameters (b1p, b2p,...).
        Any other value uses the standard Eulerian basis (b1, b2, ...).
    b3_coev : bool
        Fix b3 to its co-evolution value.
    sigma8_fid : float, optional
        Fiducial sigma_8(z_eff), used as prior centre in the physical basis.

    Returns
    -------
    params : dict[str, dict]
        Maps parameter name to a dict of keyword arguments accepted by
        :meth:`Parameter.update` (e.g. ``{'fixed': True}`` or ``{'prior': {...}}``).
    """
    params = {}
    if model == 'bao':
        tracer = get_simple_tracer(tracer)
        recon = bool(prior_basis)
        if tracer == 'BGS':
            sigmapar, sigmaper = 10., 6.5
            if recon: sigmapar, sigmaper = 8., 3.
        elif tracer == 'LRG':
            sigmapar, sigmaper = 9., 4.5
            if recon: sigmapar, sigmaper = 6., 3.
        elif tracer == 'LRG+ELG':
            sigmapar, sigmaper = 9., 4.5
            if recon: sigmapar, sigmaper = 6., 3.
        elif tracer == 'ELG':
            sigmapar, sigmaper = 8.5, 4.5
            if recon: sigmapar, sigmaper = 6., 3.
        elif tracer == 'QSO':
            sigmapar, sigmaper = 9., 3.5
            if recon: sigmapar, sigmaper = 6., 3.
        sigmas = {'sigmas': (2., 2.), 'sigmapar': (sigmapar, 2.), 'sigmaper': (sigmaper, 1.)}
        for name, value in sigmas.items():
            params[name] = {'prior': {'dist': 'norm', 'loc': value[0], 'scale': value[1], 'limits': [0., 20.]}}
        return params

    scale_eft = 12.5
    scale_sn0 = 2.0
    scale_sn2 = 5.0

    if prior_basis in ['physical', 'physical_aap', 'tcm_chudaykin_aap']:
        # ── Bias parameters ───────────────────────────────────────────────
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2. / 7. * sigma8_fid**2, 'scale': 5}}
        if 'mesh2_spectrum' in stat:
            if b3_coev:
                params['b3p'] = {'fixed': True}
            else:
                params['b3p'] = {'prior': {'dist': 'norm', 'loc': 23. / 42. * sigma8_fid**4, 'scale': sigma8_fid**4},
                                 'fixed': False}
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2p']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_pp'] = {'fixed': True}
            else:
                params['X_FoG_pp'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            params['c1p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['c2p']    = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
            params['Pshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            params['Bshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}

    else:
        # ── Bias parameters (standard Eulerian basis) ─────────────────────
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if 'mesh2_spectrum' in stat:
            if b3_coev:
                params['b3'] = {'fixed': True}
            else:
                params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}, 'fixed': False}
            # ── PS counter-terms and shot noise ───────────────────────────────
            for n in [0, 2, 4]:
                params[f'alpha{n:d}'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_eft}}
            params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn0}}
            params['sn2']  = {'prior': {'dist': 'norm', 'loc': 0, 'scale': scale_sn2}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_p'] = {'fixed': True}
            else:
                params['X_FoG_p'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
        elif 'mesh3_spectrum' in stat:
            # ── BS stochastic parameters (only for bs / joint) ────────────────
            shotnoise = 1 / 0.0002118763
            params['c1']    = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6 * 4}}
            params['c2']    = {'prior': {'dist': 'norm', 'loc': 0,    'scale': 4}}
            params['Pshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': shotnoise * 4}}
            # ── FoG damping ───────────────────────────────────────────────────
            if 'EFT' in model.upper():
                params['X_FoG_bp'] = {'fixed': True}
            else:
                params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
    return params


def get_theory(stat: str, theory_options: dict, cosmology: object=None, data_attrs: dict=None, data=None):
    """
    Return a configured theory desilike calculator for the requested statistic.

    Parameters
    ----------
    stat : str
        Statistic name, e.g. 'mesh2_spectrum' or 'mesh3_spectrum'.
    theory_options : dict
        Theory options dict containing at least 'model' and possibly other keys.
    cosmology : Cosmoprimo
        Cosmology calculator.
    data_attrs : dict
        Data attributes ('z', 'recon_mode', 'recon_smoothing_radius', 'tracers', ...).

    Returns
    -------
    theory : BaseCalculator
        Initialized theory object from desilike for the requested statistic.
    """
    from desilike.theories.galaxy_clustering import (DirectPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, BAOPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles,
    FOLPSv2TracerPowerSpectrumMultipoles, FOLPSv2TracerBispectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles)
    theory_options = dict(theory_options)
    fiducial = get_fiducial()
    template = None
    theory_options.setdefault('cosmology', {'template': 'direct'})
    cosmology_options = theory_options['cosmology']
    z = data_attrs['z']
    if cosmology_options['template'] == 'direct':
        template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmology, z=z)
    elif cosmology_options['template'] == 'shapefit':
        template = ShapeFitPowerSpectrumTemplate(fiducial=fiducial, z=z)
    elif cosmology_options['template'] == 'bao':
        kw = {name: cosmology_options[name] for name in ['apmode', 'now'] if name in cosmology_options}
        template = BAOPowerSpectrumTemplate(fiducial=fiducial, z=z, **kw)
    if template is None:
        raise ValueError(f'template not found for {stat} and {repr(cosmology_options["template"])}')
    theory = None
    if 'mesh2_spectrum' in stat:
        if theory_options['model'] == 'reptvelocileptors':
            theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, **theory_options.get('options', {}))
        elif theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis', 'b3_coev', 'A_full']}
            theory = FOLPSv2TracerPowerSpectrumMultipoles(template=template, **kw, **theory_options.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8_z(of='delta_cb', z=z)
            params = _get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['prior_basis'], b3_coev=kw['b3_coev'], sigma8_fid=sigma8_fid) | theory_options.get('params', {})
            for name, config in params.items():
                for param in theory.init.params.select(basename=name):
                    param.update(**config)
            if theory_options['marg']:
                for param in theory.init.params.select(basename=['alpha*', 'sn*']):
                    param.update(derived='.auto')
    elif 'mesh3_spectrum' in stat:
        if theory_options['model'] in ['folpsD', 'folpsEFT']:
            kw = {name: theory_options[name] for name in ['damping', 'prior_basis']}
            theory = FOLPSv2TracerBispectrumMultipoles(template=template, **kw, **theory_options.get('options', {}))
            sigma8_fid = fiducial.get_fourier().sigma8_z(of='delta_cb', z=z)
            params = _get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['prior_basis'], sigma8_fid=sigma8_fid) | theory_options.get('params', {})
            for name, config in params.items():
                for param in theory.init.params.select(basename=name):
                    param.update(**config)
    elif 'recon_particle2_correlation' in stat:
        kw = {name: np.asarray(data_attrs.get(f'recon_{name}', None)).flat[0] for name in ['mode', 'smoothing_radius']}
        kw = kw | {name: theory_options[name] for name in kw if name in theory_options}
        if kw['mode'] is None: kw['mode'] = ''  # no reconstruction
        kw['broadband'] = theory_options.get('broadband', 'pcs2')
        params = _get_default_theory_nuisance_priors(theory_options['model'], stat, prior_basis=kw['mode'], tracer=data_attrs['tracers'][0]) | theory_options.get('params', {})
        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, **kw)
        for name, config in params.items():
            for param in theory.init.params.select(basename=name):
                param.update(**config, fixed=('prior' not in config))
        ells = getattr(data, 'ells', [0, 2, 4])
        for ell in [0, 2, 4]:
            if ell not in ells:
                for param in theory.init.params.select(basename=f'*l{ell:d}_*'):
                    param.update(fixed=True)
        if len(ells) <= 1:
            theory.init.params['dbeta'].update(fixed=True)
        if False: #theory_options['marg']:
            for param in theory.init.params.select(basename=['al*', 'bl*']):
                param.update(derived='.auto')
    if theory is None:
        raise ValueError(f'theory not found for {stat} and {repr(theory_options)}')
    return theory


def pack_stats(stats, **labels):
    """
    Pack a list of stat-like objects into a single :class:`types.ObservableTree` or :class:`types.WindowMatrix`.

    Parameters
    ----------
    stats : list[ObservableLike, WindowMatrix]
        List of statistics objects to pack.
    labels : mapping
        Labels to attach to the resulting container.
        E.g. ``observables=['spectrum2', 'spectrum3'], tracers=[('LRG', 'LRG'), ('LRG', 'LRG', 'LRG')]``
        for the combined power spectrum and bispectrum.

    Returns
    -------
    Packed types.ObservableTree or types.WindowMatrix
    """
    if isinstance(stats[0], types.ObservableLike):
        return types.ObservableTree(stats, **labels)
    elif isinstance(stats[0], types.WindowMatrix):
        windows = stats
        values = [window.value() for window in windows]
        observables = [window.observable for window in windows]
        theories = [window.theory for window in windows]
        return types.WindowMatrix(
            value=sp.linalg.block_diag(*values),
            observable=pack_stats(observables, **labels),
            theory=pack_stats(theories, **labels),
        )
    else:
        raise ValueError(f'unrecognized type {stats[0]}')


def unpack_stats(stats):
    """
    Unpack packed stats structures into individual windows/observables.

    Parameters
    ----------
    stats : types.ObservableLike | types.WindowMatrix | types.GaussianLikelihood
        If ObservableLike, returns a list of observables
        If WindowMatrix, returns a list of window matrices
        If GaussianLikelihood, returns a tuple[list of observables, list of window matrices, covariance]

    Returns
    -------
    Unpacked statistics.
    """
    if isinstance(stats, types.ObservableLike):
        return stats.flatten(level=1)  # iter over labels
    elif isinstance(stats, types.WindowMatrix):
        window = stats
        windows = []
        for label in window.observable.labels(level=1):
            windows.append(window.at.observable.get(**label).at.theory.get(**label))
        return windows
    elif isinstance(stats, types.GaussianLikelihood):
        likelihood = stats
        return (unpack_stats(likelihood.observable), unpack_stats(likelihood.window), likelihood.covariance)


def combine_covariances(covariances, observable):
    """Combine input covariances into a large one, for observable."""
    olabels = observable.labels(level=1)
    nblocks = len(olabels)
    value = [[None for i in range(nblocks)] for i in range(nblocks)]
    observables = [None for i in range(nblocks)]
    for ilabel1, ilabel2 in itertools.product(range(nblocks), repeat=2):
        label1, label2 = (olabels[ilabel] for ilabel in [ilabel1, ilabel2])
        block = None
        for covariance in covariances:
            clabels = covariance.observable.labels(level=1)
            csizes = list(covariance.observable.sizes(level=1))
            cumsizes = np.cumsum([0] + csizes)
            if label1 in clabels and label2 in clabels:
                i1, i2 = clabels.index(label1), clabels.index(label2)
                block = covariance.value()[cumsizes[i1]:cumsizes[i1 + 1], cumsizes[i2]:cumsizes[i2 + 1]]
                observables[i1] = covariance.observable.get([label1])
        if block is None:
            warnings.warn(f'block {label1}, {label2} not found, assuming it is 0')
            shape = tuple(observable.get(**label).size for label in [label1, label2])
            block = np.zeros(shape)
        value[ilabel1][ilabel2] = block
    value = np.block(value)
    observable = types.join(observables)
    return types.CovarianceMatrix(observable=observable, value=value)


def _infer_effective_nparams(observables: list[dict]) -> int:
    """Infer effective free-parameter count for covariance corrections.

    Uses a fixed effective count by fit content:
      - 7 for single-stat fits (mesh2-only or mesh3-only)
      - 9 for joint mesh2+mesh3 fits
    """
    stats = {obs['stat']['kind'] for obs in observables}
    has_mesh2 = any('mesh2_spectrum' in stat for stat in stats)
    has_mesh3 = any('mesh3_spectrum' in stat for stat in stats)
    return 9 if (has_mesh2 and has_mesh3) else 7


def _get_covariance_correction_factor(covariance: types.CovarianceMatrix,
                                      observables: list[dict],
                                      covariance_options: dict,
                                      default_corrections=('hartlap', 'percival')):
    """Return multiplicative covariance correction factor and per-term metadata."""
    corrections = covariance_options.get('corrections', default_corrections)
    if isinstance(corrections, str):
        corrections = [corrections]
    corrections = [str(corr).lower() for corr in (corrections or [])]

    factor = 1.
    nbins = int(covariance.value().shape[0])
    nobs = covariance.attrs['nobs']
    metadata = {'nbins': nbins, 'corrections': tuple(corrections)}
    if nobs is None:  # analytic covariance matrix
        return factor, metadata | dict(corrections=tuple())

    nobs = int(nobs)
    metadata.update(nobs=nobs)

    if 'hartlap' in corrections:
        hartlap = get_hartlap2007_factor(nobs, nbins)
        factor /= hartlap
        metadata['hartlap_factor'] = float(hartlap)

    if 'percival' in corrections:
        nparams = covariance_options.get('nparams', None)
        if nparams is None:
            nparams = _infer_effective_nparams(observables)
        percival = get_percival2014_factor(nobs, nbins, nparams)
        factor *= percival
        metadata['percival_factor'] = float(percival)
        metadata['nparams'] = int(nparams)

    return factor, metadata


@default_mpicomm
def get_stats(observables_options: list[dict], covariance_options: dict=None, unpack: bool=False,
              get_stats_fn=clustering_tools.get_stats_fn, cache_dir: str | Path=None, cache_mode: str='rw', mpicomm=None):
    """
    Load and assemble measurement products (data, windows, covariance).

    This function:
      - reads per-statistic measurement files determined by `observables`;
      - optionally caches the assembled likelihood;
      - constructs mock-based covariance from available mock files;
      - returns a :class:`types.GaussianLikelihood` (or unpacked components if requested).

    Parameters
    ----------
    observables_options : list[dict]
        List of observable option dicts describing which stats to load.
    covariance_options : dict or None
        Options used to locate covariance/mock files.
    unpack : bool, optional
        If ``True`` return unpacked (data, windows, covariance) rather than a :class:`types.GaussianLikelihood`.
    get_stats_fn : callable
        Function used to locate stats files.
    cache_dir : str or Path, optional
        Directory to use for caching assembled likelihoods.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.

    Returns
    -------
    types.GaussianLikelihood or tuple
    """
    covariance_options = covariance_options or {}

    if cache_dir is not None:
        cache_dir = Path(cache_dir) / 'prepared_stats'
    read_cache = cache_dir is not None and 'r' in cache_mode
    write_cache = cache_dir is not None and 'w' in cache_mode

    def get_cache_fn(kind, kwargs):
        if cache_dir is None:
            return None
        _full_options = {'observables': [{name: dict(observable_options[name]) for name in ['stat', 'catalog']} for observable_options in observables_options], 'covariance': covariance_options}
        _str_from_options = str_from_likelihood_options(_full_options, level={'stat': 1, 'catalog': 2, 'covariance': 1})
        _hash = _hash_options(_full_options | kwargs)
        return cache_dir / f'{kind}_{_str_from_options}-{_hash}.h5'

    def get_from_cache(cache_fn):
        if cache_fn is None or not read_cache:
            return None
        stats = None
        if all(mpicomm.allgather(cache_fn.exists())):
            logger.info(f'Reading cached stats {cache_fn}.')
            stats = types.read(cache_fn)
        return mpicomm.bcast(stats, root=0)

    # Helper: iterate over (stat, tracer) combinations
    def iter_stat_tracer_combinations(observables_options, **kwargs):
        """
        Yield (stat, labels, file_kwargs, observable_options) for each requested observable.

        Compact helper for iterating the user-provided observables and producing file kwargs
        and labeling information used when reading files.
        """
        for observable_options in observables_options:
            stat = observable_options['stat']['kind']
            tracers = _make_tuple(observable_options['catalog']['tracer'])
            version = observable_options['catalog'].get('version', None)
            full_tracer = get_full_tracer(tracers, version=version)
            nfields = 3 if 'mesh3' in stat else 2
            simple_tracers = get_simple_tracer(tracers)
            simple_tracers += (simple_tracers[-1],) * (nfields - len(simple_tracers))
            labels = {
                'observables': get_simple_stats(stat),
                'tracers': simple_tracers,
            }
            kw = {}
            if nfields == 3:
                kw['basis'] = observable_options['stat']['basis']
            file_kw = kw | observable_options['catalog'] | {'tracer': full_tracer} | kwargs
            yield stat, labels, file_kw, dict(observable_options)

    def _with_project(observable: types.ObservableTree):
        return hasattr(observable, 'project')

    def _apply_project(observable: types.ObservableTree, select: list=None):
        # Project correlation function
        data, windows = [], []
        for _select in select:
            _select = dict(_select)
            ells = [_select.pop('ells')]
            correlation = observable
            RR = correlation.get('RR')
            for coord_name, limits in _select.items():
                if len(limits) == 3:  # apply binning only
                    step = limits[2]
                    edge = correlation.edges(coord_name)[0]
                    rebin = int(np.rint(np.mean(step / (edge[..., 1] - edge[..., 0]))) + 0.5)
                    correlation = correlation.select(**{coord_name: slice(0, None, rebin)})
                #correlation = correlation.select(**{coord_name: tuple(limits[:2])})
            pole, window = correlation.project(ells=ells, kw_window=dict(RR=RR))
            data.append(pole)
            windows.append(window)
        data = types.join(data)
        window = types.WindowMatrix(value=np.concatenate([window.value() for window in windows], axis=0),
                                    observable=types.join([window.observable for window in windows]),
                                    theory=windows[0].theory,
                                    attrs=windows[0].attrs)
        return data, window
 
    def _apply_select(observable: types.ObservableTree, select: list=None):
        """
        Apply a selection (k-range, ell selection) to an observable.

        The selection dict keys are multipoles (ell) and values are slice-like
        specifications or (min, max, [step]) tuples.
        """
        if select is None:
            return observable
        if callable(select):  # custom rebinning
            return select(observable)
        labels = []
        for _select in select:
            _select = dict(_select)
            keys = observable.labels(return_type='keys')
            label = {}
            for key in keys:
                if key in _select:
                    label[key] = _select.pop(key)
            labels.append(label)
            pole = observable.get(**label)
            for coord_name, limits in _select.items():
                if len(limits) == 3:
                    step = limits[2]
                    edge = pole.edges(coord_name)[0]
                    rebin = int(np.rint(np.mean(step / (edge[..., 1] - edge[..., 0]))) + 0.5)
                    pole = pole.select(**{coord_name: slice(0, None, rebin)})
                pole = pole.select(**{coord_name: tuple(limits[:2])})
            observable = observable.at(**label).replace(pole)
        observable = observable.get(labels)
        return observable

    def _get_mock_stats_fn(stat, file_kw):
        stats_dir = Path(file_kw.pop('stats_dir'))
        version = file_kw.pop('version', None)
        if isinstance(version, str) and version.startswith('ezmock'):
            tracer = get_simple_tracer(_make_tuple(file_kw['tracer']))
            tracer = tracer[0] if isinstance(tracer, tuple) else tracer
            zsnap = float2str(file_kw['zsnap'], 3, 3)
            imock = file_kw['imock']
            kind = {'mesh2_spectrum': 'mesh2_spectrum_poles'}.get(stat, stat)
            if 'mesh3' in kind:
                basis = file_kw.get('basis', None)
                basis = f'_{basis}' if basis else ''
                kind = f'mesh3_spectrum{basis}_poles'
            return stats_dir / version / f'{kind}_{tracer}_z{zsnap}_{imock}.h5'
        def _has_existing(fn):
            if isinstance(fn, list):
                return len(fn) > 0
            return fn.exists()
        base_fn = get_stats_fn(kind=stat, stats_dir=stats_dir, version=version, **file_kw)
        if _has_existing(base_fn):
            return base_fn
        project_fn = None
        if version is not None and file_kw.get('imock', None) is not None:
            project_fn = get_stats_fn(kind=stat, stats_dir=stats_dir, project=version, **file_kw)
            if _has_existing(project_fn):
                return project_fn
        if version is None:
            return base_fn
        alt_fn = get_stats_fn(kind=stat, stats_dir=stats_dir, **file_kw)
        return alt_fn

    def _format_log_fns(fns):
        if not isinstance(fns, list):
            return str(fns)
        if not fns:
            return '<no files>'
        if len(fns) <= 1:
            return str(fns[0])
        fns = [str(fn) for fn in fns]
        prefix = os.path.commonprefix(fns)
        suffix = os.path.commonprefix([fn[::-1] for fn in fns])[::-1]
        if prefix or suffix:
            return f'{prefix}*{suffix}'
        return '*'

    def _format_missing_data_context(stat, file_kw, fns):
        fields = ['stats_dir', 'version', 'tracer', 'zrange', 'region', 'weight', 'imock']
        context = ', '.join(f'{name}={file_kw[name]!r}' for name in fields if name in file_kw)
        return f'No measurement files found for {stat} ({context}); resolved lookup: {_format_log_fns(fns)}'

    # Loading data, window
    all_data_fns, all_imocks, joint_labels, selects = [], [], {'observables': [], 'tracers': []}, []
    for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
        file_kw = {'imock': None} | file_kw
        all_imocks.append(file_kw['imock'])
        fn = _get_mock_stats_fn(stat, file_kw) if 'stats_dir' in file_kw else get_stats_fn(kind=stat, **file_kw)
        if not isinstance(fn, list): fn = [fn]
        all_data_fns.append(fn)
        for name in joint_labels:
            joint_labels[name].append(labels[name])
        selects.append(kw['stat'].get('select', None))
    cache_data_fn = get_cache_fn('data', dict(imocks=all_imocks))
    cache_window_fn = get_cache_fn('window', dict(imocks=all_imocks))
    data = get_from_cache(cache_data_fn)
    window = get_from_cache(cache_window_fn)
    if data is None or window is None:
        for iobs, (stat, labels, file_kw, kw) in enumerate(iter_stat_tracer_combinations(observables_options)):
            if not all_data_fns[iobs]:
                raise FileNotFoundError(_format_missing_data_context(stat, file_kw, all_data_fns[iobs]))
        if mpicomm.rank == 0:
            data, windows = [], []
            for iobs, (stat, labels, file_kw, kw) in enumerate(iter_stat_tracer_combinations(observables_options)):
                _data, _windows = [], []
                logger.info(f"Reading data vector for {stat} from {_format_log_fns(all_data_fns[iobs])}")
                for fn in all_data_fns[iobs]:
                    observable = types.read(fn)
                    if _with_project(observable):  # correlation function
                        dw = _apply_project(observable, selects[iobs])
                        _data.append(dw[0])
                        _windows.append(dw[1])
                    else:  # power spectrum
                        _data.append(observable)
                data.append(types.mean(_data))
                if _windows:
                    windows.append(_windows[0])
                else:
                    imock = file_kw.get('imock', None)
                    if imock is not None:  # FIXME
                        file_kw['imock'] = 0
                    fn = _get_mock_stats_fn(f'window_{stat}', file_kw) if 'stats_dir' in file_kw else get_stats_fn(kind=f'window_{stat}', **file_kw)
                    logger.info(f"Reading window for {stat} from {fn}")
                    windows.append(types.read(fn))
            # Join mesh2_spectrum, mesh3_spectrum, etc.
            data = pack_stats(data, **joint_labels)
            window = pack_stats(windows, **joint_labels)
        data, window = mpicomm.bcast((data, window), root=0)
    if write_cache:
        write_stats(cache_data_fn, data)
        write_stats(cache_window_fn, window)
    for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
        leaf = _apply_select(data.get(**labels), select=kw['stat'].get('select', None))
        data = data.at(**labels).replace(leaf)
        window = window.at.observable.at(**labels).match(data.get(**labels))
    # Analytic covariances
    if covariance_options['source'] in ['jaxpower', 'rascalc']:
        # WARNING: not tested yet!
        full_tracers = []
        for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
            file_kw = file_kw | covariance_options
            imock = file_kw.get('imock', None)
            if imock is not None:  # FIXME
                file_kw['imock'] = 0
            full_tracers.append(file_kw['tracer'] + (file_kw['tracer'][-1],) * (len(labels['tracers']) - len(file_kw['tracer'])))
        tracers = sorted({t for tpl in full_tracers for t in tpl})
        all_combinations = []
        for tpl in full_tracers:
            n = len(tpl)
            all_combinations.extend(itertools.product(tracers, repeat=n))
        all_combinations = list(dict.fromkeys(all_combinations))  # remove duplicates
        covariances = []
        # Query all possible cross-covariances
        # FIXME if there are 3pt-covariances
        source = covariance_options['source']
        # FIXME
        if source == 'jaxpower': source = ''
        else: source = f'_{source}'
        for tracers in all_combinations:
            if all(tracer == tracers[0] for tracer in tracers):
                tracers = tracers[0]
            fn = get_stats_fn(kind=f'covariance_{stat}' + source, **(file_kw | dict(tracer=tracers)))
            if fn.exists():
                logger.info(f"Reading covariance for {stat} from {_format_log_fns(fn)}")
                covariances.append(types.read(fn))
        if not covariances:
            raise ValueError('no covariances found')
        covariance = combine_covariances(covariances, data)
        covariance.attrs['nobs'] = None
    elif covariance_options['source'] == 'mock':
        # Mock-based covariance
        all_fns = []
        covariance_log_patterns = []
        all_imocks = None
        for stat, labels, file_kw, kw in iter_stat_tracer_combinations(observables_options):
            file_kw = file_kw | {'imock': '*'} | covariance_options
            file_kw['tracer'] = get_full_tracer(get_simple_tracer(file_kw['tracer']), version=file_kw['version'])
            imocks = file_kw.pop('imock')
            if imocks == '*':
                imocks = list(range(2001))
            if all_imocks is None:
                all_imocks = list(imocks)
            else:
                all_imocks = [imock for imock in all_imocks if imock in imocks]
            stat_fns = [_get_mock_stats_fn(stat, file_kw | {'imock': imock}) for imock in imocks]
            all_fns.append(stat_fns)
            covariance_log_patterns.append((stat, _format_log_fns(stat_fns)))
        all_fns = list(zip(*all_fns, strict=True))  # get a list of list of file names
        ifns_exists = []
        if mpicomm.rank == 0:
            for stat, pattern in covariance_log_patterns:
                logger.info(f"Looking for covariance mocks for {stat} at {pattern}")
            for ifn, fns in enumerate(all_fns):
                if all(fn.exists() for fn in fns):
                    ifns_exists.append(ifn)
        ifns_exists = mpicomm.bcast(ifns_exists, root=0)
        imocks_exists = [all_imocks[ifn] for ifn in ifns_exists]
        cache_fn = get_cache_fn('covariance', dict(imocks=imocks_exists))
        covariance = get_from_cache(cache_fn)
        if covariance is None:
            mocks = []
            if mpicomm.rank == 0:
                covariance_read_fns = [all_fns[ifn] for ifn in ifns_exists]
                logger.info(f"Reading covariance for {len(covariance_read_fns)} mock realizations from "
                            f"{_format_log_fns(covariance_read_fns)}")
                for ifn in ifns_exists:
                    # Join mesh2_spectrum, mesh3_spectrum, etc.
                    observables = [types.read(fn) for fn in all_fns[ifn]]
                    observables = [_apply_project(observable, select)[0] if _with_project(observable) else _apply_select(observable, select)
                                   for observable, select in zip(observables, selects)]
                    mock = types.ObservableTree(observables, **joint_labels)
                    mocks.append(mock)
                covariance = types.cov(mocks)
                covariance.attrs['nobs'] = len(mocks)
            covariance = mpicomm.bcast(covariance, root=0)
        if cache_fn is not None:
            write_stats(cache_fn, covariance)

    covariance = covariance.at.observable.match(data)

    factor, metadata = _get_covariance_correction_factor(covariance, observables_options, covariance_options)
    if factor != 1.:
        covariance = covariance.clone(value=covariance.value() * factor)
    covariance.attrs['covariance_correction_factor'] = float(factor)
    for name, value in metadata.items():
        covariance.attrs[name] = value
    if metadata['corrections']:
        info = f"Applied covariance corrections {metadata['corrections']} with factor {factor:.6f}"
        if 'hartlap_factor' in metadata:
            info += f", hartlap={metadata['hartlap_factor']:.6f}"
        if 'percival_factor' in metadata:
            info += f", percival={metadata['percival_factor']:.6f}, nparams={metadata['nparams']}"
        if mpicomm.rank == 0:
            logger.info(info)

    likelihood = types.GaussianLikelihood(
        observable=data,
        window=window,
        covariance=covariance,
    )

    if unpack:
        return unpack_stats(likelihood)
    return likelihood


@default_mpicomm
def get_single_likelihood(likelihood_options, stats: types.GaussianLikelihood=None,
                          cosmology_options: dict=None, get_stats_fn=clustering_tools.get_stats_fn,
                          get_theory=get_theory, cache_dir:str | Path=None, cache_mode: str='rw', mpicomm=None):
    """
    Build a single :mod:`desilike` Gaussian likelihood from provided options.

    Parameters
    ----------
    likelihood_options : dict
        Options containing 'observables' list and 'covariance' dict.
    stats : types.GaussianLikelihood or None
        Preloaded measurements (if ``None`` they will be loaded via :func:`get_stats`).
    cosmology_options : optional
        Cosmology options or object or :class:`desilike.theories.Cosmoprimo`.
    get_stats_fn : callable, optional
        Function to locate measurement files.
    cache_dir : str | Path, optional
        Directory used for caching pre-computed emulators.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.

    Returns
    -------
    ObservablesGaussianLikelihood
    """
    from desilike.observables.galaxy_clustering import TracerSpectrum2PolesObservable, TracerSpectrum3PolesObservable, TracerCorrelation2PolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    # likelihood_options: {'observables': [observable_options], 'covariance': {}}
    observables_options = likelihood_options['observables']
    covariance_options = likelihood_options.get('covariance', {})
    cosmology = get_cosmology(cosmology_options)
    if stats is None:
        stats = get_stats(observables_options, covariance_options=covariance_options, unpack=False, get_stats_fn=get_stats_fn, cache_dir=cache_dir, cache_mode=cache_mode)
    data, windows, covariance = unpack_stats(stats)
    labels = covariance.observable.labels(level=1)
    observables = []
    for observable_options, data, window, label in zip(observables_options, data, windows, labels, strict=True):
        stat = observable_options['stat']['kind']
        if 'mesh2_spectrum' in stat:
            cls = TracerSpectrum2PolesObservable
        elif 'mesh3_spectrum' in stat:
            cls = TracerSpectrum3PolesObservable
        elif 'particle2_correlation' in stat:
            cls = TracerCorrelation2PolesObservable
        else:
            raise NotImplementedError(stat)
        data_attrs = dict(data.attrs) | label
        for label, pole in window.observable.items(level=None):
            data_attrs['z'] = pole.attrs['zeff']
        theory = get_theory(stat, theory_options=observable_options['theory'], cosmology=cosmology, data_attrs=data_attrs, data=data)
        namespace = _str_from_observable_options(
            observable_options, level={'catalog': 1, 'stat': 0, 'theory': 0, 'covariance': 0})
        theory_params = theory.init.params
        observable = cls(data=data, window=window, theory=theory)
        observable()
        if observable_options['emulator']['name']:
            assert cache_dir is not None, 'cache_dir must be provided for emulator'
            read_cache = cache_dir is not None and 'r' in cache_mode
            write_cache = cache_dir is not None and 'w' in cache_mode
            cache_dir = Path(cache_dir)
            _hash = _hash_options({name: observable_options[name] for name in ['theory', 'catalog']})
            _str_cosmology = str_from_cosmology_options(observable_options['theory']['cosmology'], level=100)
            _str_cosmology += '_' + observable_options['emulator']['name']
            _str_theory = _str_from_observable_options(observable_options, level={'theory': 100, 'catalog': 2})
            cache_fn = cache_dir / f'emulator_{_str_cosmology}' / f'emulator_{_str_theory}_{_hash}.npy'
            from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
            emulated_pt = None
            if read_cache and cache_fn.exists():
                logger.info(f'Reading cached emulator {cache_fn}')
                emulated_pt = EmulatedCalculator.load(cache_fn)
            else:
                logger.info(f'Fitting emulator {cache_fn}')
                emulator = Emulator(
                    theory.pt,
                    engine=TaylorEmulatorEngine(method='finite', order=observable_options['emulator'].get('order', 3)),
                )
                emulator.set_samples()
                emulator.fit()
                emulated_pt = emulator.to_calculator()
                if write_cache:
                    emulated_pt.save(cache_fn)
            theory.init.update(pt=emulated_pt)
            theory.init.params.update(theory_params)
            for param in theory.init.params:
                param.update(namespace=namespace)
        observables.append(observable)
    return ObservablesGaussianLikelihood(observables, covariance=covariance.value())


def get_likelihood(likelihoods_options: dict | list[dict], cosmology_options: dict=None, get_stats_fn=clustering_tools.get_stats_fn,
                   get_theory=get_theory, cache_dir:str | Path=None, cache_mode: str='rw'):
    """
    Build a desilike :class:`SumLikelihood, summed over all tracers.

    Parameters
    ----------
    likelihoods_options : dict, list[dict]
        List of options {'observables': [observable_options, ...], 'covariance': {}}.
    cosmology_options : optional
        Cosmology options or object or :class:`desilike.theories.Cosmoprimo`.
    get_stats_fn : callable, optional
        Function to locate measurement files.
    cache_dir : str | Path, optional
        Directory used for caching pre-computed emulators.
    cache_mode : str, optional
        'rw' for read/write; 'r' for read-only.

    Returns
    -------
    SumLikelihood
    """
    from desilike.likelihoods import SumLikelihood
    cosmology = get_cosmology(cosmology_options)
    if isinstance(likelihoods_options, dict):
        likelihoods_options = [likelihoods_options]
    likelihoods = [get_single_likelihood(likelihood_options, cosmology_options=cosmology,
                                         get_stats_fn=get_stats_fn, get_theory=get_theory,
                                         cache_dir=cache_dir, cache_mode=cache_mode) for likelihood_options in likelihoods_options]
    return SumLikelihood(likelihoods)


def get_sampler_cls(name):
    """Return sampler class."""
    from desilike.samplers import EmceeSampler
    translate = {'emcee': EmceeSampler}
    return translate[name.lower()]


def get_profiler_cls(name):
    """Return profiler class."""
    from desilike.profilers import MinuitProfiler
    translate = {'minuit': MinuitProfiler}
    return translate[name.lower()]
    

def propose_fiducial_observable_options(stat, tracer=None, zrange=None):
    """Propose fiducial fitting options for given statistics and tracer."""
    propose_fiducial = {'stat': {'kind': stat},
                        'catalog': {'weight': 'default-FKP'},
                        'theory': {},
                        'emulator': {'name': 'taylor', 'order': 3},
                        'window': {}}
    propose_stat = {'mesh2_spectrum': {'select': [{'ells': ell, 'k': [0.02, 0.2, 0.005]} for ell in [0, 2]]},
                    'mesh3_spectrum': {'select': [{'ells': (0, 0, 0), 'k': [0.02, 0.12, 0.005]}, {'ells': (2, 0, 2), 'k': [0.02, 0.08, 0.005]}],
                                        'basis': 'sugiyama-diagonal'},
                   'recon_particle2_correlation': {'select': [{'ells': ell, 's': [60., 150., 4.]} for ell in [0, 2]]}}
    base_full_shape_theory = {'model': 'folpsD', 'prior_basis': 'physical_aap', 'damping': 'lor', 'marg': True}
    base_bao_theory = {'model': 'bao', 'broadband': 'pcs2', 'marg': True}
    propose_theory = {'mesh2_spectrum': base_full_shape_theory | {'b3_coev': True, 'A_full': False},
                      'mesh3_spectrum': base_full_shape_theory | {'A_full': False},
                      'recon_particle2_correlation': base_bao_theory}
    for _stat in propose_stat:
        if _stat in stat:
            propose_fiducial['stat'].update(propose_stat[_stat])
            propose_fiducial['theory'].update(propose_theory[_stat])
    return propose_fiducial


def propose_fiducial_covariance_options():
    """Return dictionary of default covariance options."""
    return {'source': 'mock', 'version': 'holi-v1-altmtl', 'corrections': ['hartlap', 'percival']}


def propose_fiducial_cosmology_options():
    """Return dictionary of default cosmology options."""
    return {'model': 'base_ns-fixed', 'template': 'direct'}


def propose_fiducial_sampler_options(sampler=None):
    """Return dictionary of default sampler configuration."""
    if sampler is None:
        sampler = 'emcee'
    fiducial_options = {'sampler': sampler, 'init': {}, 'run': {'check': {'max_eigen_gr': 0.03}}, 'nchains': 1}
    return fiducial_options


def propose_fiducial_profiler_options(profiler=None):
    """Return dictionary of default profiler configuration."""
    if profiler is None:
        profiler = 'minuit'
    fiducial_options = {'profiler': profiler, 'init': {}, 'maximize': {}}
    return fiducial_options


def fill_fiducial_observable_options(options):
    """Fill missing observable options with fiducial values."""
    options = dict(options)
    stat = options['stat']['kind']
    tracer = options['catalog'].get('tracer', None)
    zrange = options['catalog'].get('zrange', None)
    fiducial_options = propose_fiducial_observable_options(stat, tracer, zrange)
    options = fiducial_options | options
    for key, value in fiducial_options.items():
        options[key] = value | options[key]
    return options


def fill_fiducial_likelihood_options(options):
    """Fill missing likelihood options with fiducial values."""
    if isinstance(options, dict):
        options = dict(options)
        options['observables'] = [fill_fiducial_observable_options(options) for options in options['observables']]
        options['covariance'] = propose_fiducial_covariance_options() | (options.get('covariance', {}) or {})
        return options
    return type(options)(fill_fiducial_likelihood_options(opts) for opts in options)


def fill_fiducial_options(options):
    """Fill missing options with fiducial values."""
    options = dict(options)
    options['cosmology'] = propose_fiducial_cosmology_options() | {'template': 'direct'} | options.get('cosmology', {})
    likelihoods = options.get('likelihoods', None)
    if likelihoods is not None:
        options['likelihoods'] = fill_fiducial_likelihood_options(options['likelihoods'])
        # Add cosmology arguments to each observable
        for likelihood_options in options['likelihoods']:
            for observable_options in likelihood_options['observables']:
                observable_options['theory']['cosmology'] = options['cosmology']
    for name in ['sampler', 'profiler']:
        options.setdefault(name, {})
        options[name] = globals()[f'propose_fiducial_{name}_options'](options[name].get(name)) | options[name]
    return options


def generate_likelihood_options_helper(stats=('mesh2_spectrum', 'mesh3_spectrum'),
                                       tracer='LRG', zrange=(0.4, 0.6), region='GCcomb',
                                       version='abacus-2ndgen-complete',
                                       covariance='holi-v1-altmtl',
                                       stats_dir=Path('/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe'),
                                       emulator=True):
    """
    Convenience helper that builds a minimal dictionary of likelihood options.

    Parameters
    ----------
    stats : list
        List of statistics in the joint likelihood, from ['mesh2_spectrum', 'mesh3_spectrum']
    tracer : str, tuple
        Tracers to fit.
    zrange : tuple
        Redshift range.
    region : str
        Sky region.
    version : str
        Version of data to use.
    covariance : str
        Version of covariance mocks to use.

    Returns
    -------
    likelihood_options : dict
        Dictionary with keys ['observables', 'covariance'].
        'covariance' is a dictionary specifying how to construct the covariance matrix.
        'observables' contains a list of dictionary (one for each observable), with keys:
        {'stat': {'kind': ..., 'basis': ..., 'select': [...]}, 'catalog': {'version':, ...}, 'theory': {'model': ...}, 'window': {}}

    """
    if isinstance(stats, str):
        stats = [stats]
    observables = []
    tracer, zrange = get_full_tracer_zrange(tracer)
    for stat in stats:
        catalog = {'version': version, 'tracer': tracer, 'zrange': zrange, 'region': region, 'stats_dir': stats_dir}
        if 'data' not in version:
            catalog['imock'] = '*'  # read all available mocks
        observable_options = {'stat': {'kind': stat}, 'catalog': catalog}
        if emulator is False: emulator_options = {'name': ''}
        elif emulator is True: emulator_options = {}
        else: emulator_options = dict(emulator)
        observable_options['emulator'] = emulator_options
        observables.append(observable_options)
    covariance = {'version': covariance, 'stats_dir': stats_dir}
    return fill_fiducial_likelihood_options({'observables': observables, 'covariance': covariance})


def generate_box_likelihood_options_helper(
        stats=('mesh2_spectrum',),
        tracer='LRG',
        zsnap=0.800,
        cosmo='000',
        hod='',
        los='z',
        version='abacus-2ndgen',
        covariance_version='ezmock-dr1',
        stats_dir=Path('/dvs_ro/cfs/cdirs/desicollab/mocks/cai/LSS/DA2/mocks/desipipe/box'),
        covariance_stats_dir=None,
        emulator=True):
    """
    Convenience helper that builds a dictionary of likelihood options for cubic box mocks.

    Parameters
    ----------
    stats : list
        Statistics to fit, e.g. ['mesh2_spectrum', 'mesh3_spectrum'].
    tracer : str
        Simple tracer name ('LRG', 'ELG', 'QSO').
    zsnap : float
        Box snapshot redshift (e.g. 0.800 for LRG mid-z bin).
    cosmo : str
        Cosmology label (e.g. '000').
    hod : str
        HOD flavor string (empty string for abacus-2ndgen baseline).
    los : str
        Line-of-sight direction ('x', 'y', or 'z').
    version : str
        Version subdirectory under stats_dir for the data vector (e.g. 'abacus-2ndgen').
    covariance_version : str
        Version subdirectory for the covariance mocks (e.g. 'ezmock-dr1').
    stats_dir : Path
        Base directory containing version-named subdirectories of box measurements.
    covariance_stats_dir : Path, optional
        Base directory containing version-named covariance mock measurements.
    emulator : bool or dict
        Emulator configuration.

    Returns
    -------
    likelihood_options : dict
        Dictionary with keys ['observables', 'covariance'].
    """
    if isinstance(stats, str):
        stats = [stats]
    simple_tracer = get_simple_tracer(_make_tuple(tracer))
    observables = []
    for stat in stats:
        catalog = {
            'tracer': simple_tracer,
            'zsnap': zsnap,
            'cosmo': cosmo,
            'hod': hod,
            'los': los,
            'version': version,
            'imock': '*',
            'stats_dir': stats_dir,
        }
        observable_options = {'stat': {'kind': stat}, 'catalog': catalog}
        if emulator is False: emulator_options = {'name': ''}
        elif emulator is True: emulator_options = {}
        else: emulator_options = dict(emulator)
        observable_options['emulator'] = emulator_options
        observables.append(observable_options)
    if covariance_stats_dir is None:
        covariance_stats_dir = stats_dir
    covariance = {'source': 'mock', 'version': covariance_version, 'stats_dir': covariance_stats_dir,
                  'corrections': ['hartlap', 'percival']}
    return fill_fiducial_likelihood_options({'observables': observables, 'covariance': covariance})


def get_full_tracer_zrange(tracerz=None, zrange=None):
    """
    Translate simple tracer labels, (e.g. LRG1),
    to full tracer and zrange tuples ('LRG', (0.4, 0.6)).

    Parameters
    ----------
    tracerz : str, tuple, list, None
        If None returns the mapping table. If tracerz is a string returns
        (tracer, zrange) or for compound tracer strings returns zipped tuples.

    Returns
    -------
    tracer, zrange
    """
    translate_zrange = {'BGS1': (0.1, 0.4),
                        'LRG1': (0.4, 0.6), 'LRG2': (0.6, 0.8), 'LRG3': (0.8, 1.1),
                        'ELG1': (0.8, 1.1), 'ELG2': (1.1, 1.6),
                        'QSO1': (0.8, 2.1)}
    if tracerz is None:
        return translate_zrange

    def _get_full_tracer_zrange(tracerz, zrange=zrange):
        if 'x' in tracerz:
            return list(zip(*[_get_full_tracer_zrange(t, zrange=zrange) for t in tracerz.split('x')]))
        if tracerz in translate_zrange:
            # Return tracer and z-range from translate_zrange
            tracer = tracerz[:-1]
            zrange = translate_zrange[tracerz]
        else:
            # Not in translate_zrange
            tracer = tracerz
        if zrange is None:
            raise ValueError(f'zrange not found for {tracerz}; choose one from {list(translate_zrange)}')
        return tracer, zrange

    if isinstance(tracerz, str):
        return _get_full_tracer_zrange(tracerz)
    else:  # tuple/list of tracers
        return type(tracerz)(zip(*map(_get_full_tracer_zrange, tracerz)))


def _get_level(level: int | dict=None):
    """Compact helper to normalise verbosity level for string helpers."""
    _default_level = {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0, 'cosmology': 1}
    if level is None: level = {}
    if not isinstance(level, dict):
        level = {name: level for name in _default_level}
    level = _default_level | level
    return level


def _base_type_options(options):
    """
    Recursively cast objects of input dictionary ``d`` to Python base types
    so they can be serialized by standard YAML.
    """
    def convert(v):
        if isinstance(v, dict):
            return {k: convert(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple, set, frozenset)):
            return [convert(vv) for vv in v]
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return convert(v.item())
            return [convert(vv) for vv in v.tolist()]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if v is None or isinstance(v, (bool, numbers.Number, str)):
            return v
        return str(v)
    return convert(options)


def _hash_options(options, length=8):
    """Return a short SHA-256 hash of a canonicalized options dict."""
    def _canonical(obj):
        if isinstance(obj, dict):
            return sorted((_canonical(k), _canonical(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return [_canonical(x) for x in obj]
        return obj
    s = json.dumps(_canonical(_base_type_options(options)), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:length]


def _str_from_observable_options(options: dict, level: int=None) -> str:
    """Return string identifier given input observable options, with ``level`` of details."""
    level = _get_level(level)
    out_str = []

    # First, catalog
    catalog = _unzip_catalog_options(options['catalog'])

    def _str_zrange(zrange):
        return f'z{float2str(zrange[0], prec_min=1, prec_max=5)}-{float2str(zrange[1], prec_min=1, prec_max=5)}'

    if level['catalog'] >= 1:
        translate_tracerz = get_full_tracer_zrange(tracerz=None)
        catalog_str = []
        for tracer in catalog:
            stracer = get_simple_tracer(tracer)
            catalog_options = catalog[tracer]
            found = False
            if 'zrange' in catalog_options:
                for tracerz, zrange in translate_tracerz.items():
                    if tracerz.startswith(stracer) and np.allclose(catalog_options['zrange'], zrange):
                        stracer = tracerz  # e.g. LRG1
                        found = True
                        break
            tracer_catalog_str = [stracer]
            if 'zrange' in catalog_options:
                if not found or level['catalog'] >= 2:
                    tracer_catalog_str.append(_str_zrange(catalog_options['zrange']))
            elif 'zsnap' in catalog_options:
                tracer_catalog_str.append(f'z{float2str(catalog_options["zsnap"], prec_min=1, prec_max=5)}')
            if level['catalog'] >= 3:
                tracer_catalog_str.append(catalog_options['region'])
            if level['catalog'] >= 4:
                tracer_catalog_str.append('weight-' + catalog_options['weight'])
            catalog_str.append('-'.join(tracer_catalog_str))
        out_str.append('x'.join(catalog_str))

    # Then, stat and select, e.g. S2-ell0-k-0.02-0.2-ell2-k-0.02-0.2
    translate_stat_name = {'S2': ['mesh2_spectrum'],
                           'S3': ['mesh3_spectrum'],
                           'BAOR': ['bao', 'recon'],
                           'C2R': ['particle2_correlation', 'recon']}
    stat_options = options['stat']
    stat = stat_options['kind']
    if level['stat'] >= 1:
        found = None
        for name in translate_stat_name:
            if all(t in stat for t in translate_stat_name[name]):
                found = name
                break
        if found is None:
            raise ValueError(f'could not find shot naame for {stat}')
        out_str.append(found)
    if level['stat'] >= 2:
        select_str = []
        select = stat_options.get('select', [])
        if callable(select):  # custom binning
            select_str.append(getattr(select, 'name', 'custom'))
        else:
            def _str_ell(ell):
                if isinstance(ell, (list, tuple)):
                    ell = ''.join([str(ell) for ell in ell])
                else:
                    ell = str(ell)
                return ell
                        
            for _select in select:
                _select = dict(_select)
                label = []
                for key in list(_select):
                    if key == 'ells':
                        label.append('ell' + _str_ell(_select.pop(key)))
                for coord_name, limits in _select.items():
                    prec = dict(prec_min=2, prec_max=3) if name.startswith('S') else dict(prec_min=0, prec_max=0)
                    label.append(coord_name + '-'.join(float2str(lim, **prec) for lim in limits))
                select_str.append('-'.join(label))
        select_str = '-'.join(select_str)
        out_str.append(select_str)

    if level['theory'] > 0:
        out_str.append('th')
        out_str.append(options['theory']['model'])

    return '-'.join(out_str)


def str_from_likelihood_options(likelihood_options, level: int | dict=None):
    """
    Return a compact string identifier for likelihood options.

    Parameters
    ----------
    likelihood_options : dict
        Dictionary with keys 'observables', 'covariance'.
    level : dict
        "Verbosity level". Default is {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0}.
        Increase for more details.
        Covariance level behavior:
        - > 0: include covariance version
        - >= 3: include covariance corrections and optional nparams
    """
    level = _get_level(level)
    out_str = []
    for options in likelihood_options['observables']:
        out_str.append(_str_from_observable_options(options, level=level))
    if level['covariance'] > 0:
        covariance = likelihood_options.get('covariance', {}) or {}
        covariance_str = []
        covariance_str.append(covariance.get('source', 'none'))
        covariance_str.append(covariance.get('version', 'none'))
        covariance_str = ['cov-' + '-'.join(covariance_str)]
        if level['covariance'] >= 3:
            corrections = covariance.get('corrections', None)
            if isinstance(corrections, str):
                corrections = [corrections]
            corrections = sorted(str(corr).lower() for corr in (corrections or []))
            if corrections:
                covariance_str.append('corr-' + '-'.join(corrections))
            nparams = covariance.get('nparams', None)
            if nparams is not None:
                covariance_str.append(f'nparams-{int(nparams)}')
        out_str.append('-'.join(covariance_str))
    return '+'.join(out_str)


def str_from_cosmology_options(cosmology_options: dict, level: int | dict=None):
    """
    Return a compact string identifier for cosmology options.

    Parameters
    ----------
    cosmology_options : dict
        Dictionary with keys 'model', 'template'.
    level : dict
        "Verbosity level". Default is {'cosmology': 1}.
        Increase for more details.
    """
    level = _get_level(level)
    out_str = []
    if level['cosmology'] >= 1:
        model, template = cosmology_options['model'], cosmology_options['template']
        if template.lower() == 'direct':
            out_str.append(f'cosmo-{model}')
        else:
            out_str.append(f'template-{template}')
    return '-'.join(out_str)


def str_from_options(options: dict, level: int | dict=None):
    """
    Return a compact string identifier for options.

    Parameters
    ----------
    options : dict
        Dictionary with keys 'likelihoods', 'cosmology'.
    level : dict
        "Verbosity level". Default is {'stat': 1, 'catalog': 1, 'theory': 0, 'covariance': 0, 'cosmology': 1}.
        Increase for more details.
    """
    level = _get_level(level)
    out_str = [str_from_cosmology_options(options['cosmology'], level=level)]
    out_str += [str_from_likelihood_options(likelihood_options, level=level) for likelihood_options in options['likelihoods']]
    return '_'.join(out_str)


def get_fits_fn(fits_dir=Path(os.getenv('SCRATCH', '.')) / 'fits', kind='chain', likelihoods: list=None,
                sampler: dict=None, profiler: dict=None, cosmology: dict=None, ichain: int=None,
                level=None, extra='', ext='npy'):
    """
    Construct a file path for fit outputs based on likelihood and run options.

    Parameters
    ----------
    fits_dir : str, Path
        Base directory for fit outputs.
    kind : str
        Fitting product. Options are 'chain', 'profiles', etc.
    likelihoods : list
        Likelihood options used to build the filename.
    ichain : int or None
        Optional chain index appended to filename.
    extra : str, optional
        Extra suffix to include in the path.
    ext : str, optional
        File extension. Default is 'npy'.

    Returns
    -------
    fn : Path
        Fit file name.
    """
    fits_dir = Path(fits_dir)
    options = {'likelihoods': likelihoods, 'cosmology': cosmology}
    _str_from_options = str_from_options(options, level=level)
    _hash = _hash_options(options)
    extra = f'_{extra}' if extra else ''
    ichain = f'_{ichain:d}' if ichain is not None else ''
    return fits_dir / f'{_str_from_options}-{_hash}{extra}' / f'{kind}{ichain}.{ext}'



try:
    import yaml
except ImportError:
    yaml = None


def write_options(filename, options):
    """Write options to ``filename``."""
    options = _base_type_options(options)

    class FlowList(list):
        pass
    
    def flow_list_representer(dumper, data):
        return dumper.represent_sequence(
            'tag:yaml.org,2002:seq',
            data,
            flow_style=True,
        )
    
    yaml.add_representer(FlowList, flow_list_representer)

    def mark_flow_lists(obj):
        if isinstance(obj, dict):
            return {k: mark_flow_lists(v) for k, v in obj.items()}
        if isinstance(obj, list):
            obj = [mark_flow_lists(v) for v in obj]
            # choose the lists you want inline
            if all(not isinstance(v, (dict, list)) for v in obj):
                return FlowList(obj)
            return obj
        return obj

    # To use flow style for simple lists
    options = mark_flow_lists(options)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as file:
        yaml.dump(options, file, sort_keys=False, default_flow_style=False)


def read_options(filename):
    """Read options from ``filename``."""

    class YamlLoader(yaml.SafeLoader):
        """
        *yaml* loader that correctly parses numbers.
        Taken from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number.
        """
    
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    YamlLoader.add_implicit_resolver(u'tag:yaml.org,2002:float',
                                     re.compile(u'''^(?:
                                     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                                     |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                                     |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                                     |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                                     |[-+]?\\.(?:inf|Inf|INF)
                                     |\\.(?:nan|NaN|NAN))$''', re.X),
                                     list(u'-+0123456789.'))
    
    YamlLoader.add_implicit_resolver('!none', re.compile('None$'), first='None')

    def none_constructor(loader, node):
        return None
    
    YamlLoader.add_constructor('!none', none_constructor)

    with open(filename, 'r') as file:
        return yaml.load(file, Loader=YamlLoader)
