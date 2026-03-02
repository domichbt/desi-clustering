import os
os.environ["FOLPS_BACKEND"] = "jax"
import itertools

import numpy as np
from pathlib import Path

from desilike.theories.galaxy_clustering import (
    REPTVelocileptorsTracerPowerSpectrumMultipoles,
    DirectPowerSpectrumTemplate,
)
from desilike.theories.galaxy_clustering.full_shape import (
    FOLPSv2TracerPowerSpectrumMultipoles,
    FOLPSv2TracerBispectrumMultipoles,
)
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI
from desilike import setup_logging, ParameterCollection
from desilike.samplers import EmceeSampler

import lsstypes as types
from lsstypes import ObservableTree
from clustering_statistics.tools import get_stats_fn

# ─── Module-level constants ──────────────────────────────────────────────────

TRACER_TYPES: dict = {
    'LRG1': ('LRG',     (0.4, 0.6)),
    'LRG2': ('LRG',     (0.6, 0.8)),
    'LRG3': ('LRG',     (0.8, 1.1)),
    'QSO':  ('QSO',     (0.8, 2.1)),
    'ELG':  ('ELG_LOP', (1.1, 1.6)),
}

TRACER_REDSHIFTS = {
    'BGS':  0.295,
    'LRG1': 0.5094,
    'LRG2': 0.7054,
    'LRG3': 0.9264,
    'ELG':  1.3442,
    'QSO':  1.4864,
}

TRACER_SIGMA8 = {
    'BGS':  0.69376997,
    'LRG1': 0.62404056,
    'LRG2': 0.53930063,
    'LRG3': 0.50823223,
    'ELG':  0.43197292,
    'QSO':  0.41825647,
}

_STATS_DIR = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')


# ─── Nuisance priors ─────────────────────────────────────────────────────────

def get_nuisance_priors(prior_basis, width_EFT, width_SN0, width_SN2,
                        pt_model='folpsD', b3_coev=True, sigma8_fid=None):
    """Build a :class:`ParameterCollection` of nuisance parameter priors for a
    joint P(k)+B(k) fit.

    Compared to the power-spectrum-only version, this function additionally
    sets priors on the bispectrum stochastic parameters (``c1``, ``c2``,
    ``Pshot``, ``Bshot`` in the standard basis; ``c1p``, ``c2p``, ``Pshotp``,
    ``Bshotp`` in the physical basis).

    Parameters
    ----------
    prior_basis : str
        Parametrisation basis.  ``'physical'`` or ``'physical_prior_doc'``
        uses physical bias parameters (``b1p``, ``b2p``, …); any other value
        uses the standard Eulerian basis (``b1``, ``b2``, …).
    width_EFT : float
        Standard deviation of the Gaussian prior on EFT counter-term
        parameters (``alpha0``, ``alpha2``, ``alpha4``).
    width_SN0 : float
        Standard deviation of the Gaussian prior on the shot-noise monopole
        stochastic parameter (``sn0``).
    width_SN2 : float
        Standard deviation of the Gaussian prior on the shot-noise quadrupole
        stochastic parameter (``sn2``).
    pt_model : str, optional
        Perturbation theory model tag.  When ``'EFT'``, the finger-of-god
        damping parameters are fixed.  Default ``'folpsD'``.
    b3_coev : bool, optional
        If ``True``, the third-order bias ``b3`` is fixed to its co-evolution
        value and not sampled.  Default ``True``.
    sigma8_fid : float, optional
        Fiducial :math:`\\sigma_8` at the tracer effective redshift.  Used to
        set the prior centre for ``bsp`` and ``b3p`` in the physical basis.

    Returns
    -------
    params : :class:`ParameterCollection`
        Collection of nuisance parameters with priors attached.
    """
    params = ParameterCollection()

    if prior_basis in ('physical', 'physical_prior_doc'):
        # ── Shared bias params ────────────────────────────────────────────
        params['b1p'] = {'prior': {'dist': 'uniform', 'limits': [0.1, 4]}}
        params['b2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2/7 * sigma8_fid**2, 'scale': 5}}
        if b3_coev:
            params['b3p'] = {'fixed': True}
        else:
            params['b3p'] = {
                'prior': {'dist': 'norm', 'loc': 23/42 * sigma8_fid**4, 'scale': sigma8_fid**4},
                'fixed': False,
            }
        # ── PS counter-terms ──────────────────────────────────────────────
        params['alpha0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        if pt_model == 'EFT':
            params['X_FoG_pp'] = {'fixed': True}
            params['X_FoG_bp'] = {'fixed': True}
        else:
            params['X_FoG_pp'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
            params['X_FoG_bp'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
        # ── BS-specific stochastic params ─────────────────────────────────
        params['c1p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['c2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['Pshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
        params['Bshotp'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}

    else:
        # ── Shared bias params (standard Eulerian basis) ──────────────────
        params['b1'] = {'prior': {'dist': 'uniform', 'limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist': 'uniform', 'limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'fixed': True}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}
        # ── PS counter-terms ──────────────────────────────────────────────
        params['alpha0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        if pt_model == 'EFT':
            params['X_FoG_p'] = {'fixed': True}
            params['X_FoG_b'] = {'fixed': True}
        else:
            params['X_FoG_p'] = {'prior': {'dist': 'uniform', 'limits': [0, 10]}}
            params['X_FoG_b'] = {'prior': {'dist': 'uniform', 'limits': [0, 15]}}
        # ── BS-specific stochastic params ─────────────────────────────────
        Ppoisson = 1 / 0.0002118763
        params['c1'] = {'prior': {'dist': 'norm', 'loc': 66.6, 'scale': 66.6 * 4}}
        params['c2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 4}}
        params['Pshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': Ppoisson * 4}}
        params['Bshot'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': Ppoisson * 4}}

    return params


# ─── Cosmology ───────────────────────────────────────────────────────────────

def get_cosmology():
    """Initialise a :class:`Cosmoprimo` cosmology and a DESI fiducial cosmology.

    Sets up a CLASS-based cosmology calculator with free parameters
    ``h``, ``omega_cdm``, ``omega_b``, ``logA`` (uniform priors) and
    ``n_s``, ``tau_reio`` fixed to their fiducial values.  A Gaussian prior
    is placed on ``omega_b`` informed by simulations.

    Returns
    -------
    cosmo : :class:`Cosmoprimo`
        Cosmology calculator with priors configured for sampling.
    fiducial : :class:`cosmoprimo.Cosmology`
        DESI fiducial cosmology used for template evaluation.
    """
    cosmo = Cosmoprimo(engine='class')
    cosmo.init.params['H0'] = dict(derived=True)
    cosmo.init.params['Omega_m'] = dict(derived=True)
    cosmo.init.params['sigma8_m'] = dict(derived=True)
    fiducial = DESI()

    for param in ['n_s', 'h', 'omega_cdm', 'omega_b', 'logA', 'tau_reio']:
        cosmo.params[param].update(fixed=False)
        if param == 'tau_reio':
            cosmo.params[param].update(fixed=True)
        if param == 'n_s':
            cosmo.params[param].update(fixed=True)
        if param == 'omega_b':
            cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})
        if param == 'h':
            cosmo.params[param].update(prior={'dist': 'uniform', 'limits': [0.5, 0.9]})
        if param == 'omega_cdm':
            cosmo.params[param].update(prior={'dist': 'uniform', 'limits': [0.05, 0.2]})
        if param == 'logA':
            cosmo.params[param].update(prior={'dist': 'uniform', 'limits': [2.0, 4.0]})

    return cosmo, fiducial


# ─── Theory ──────────────────────────────────────────────────────────────────

def get_theory(tracer, fiducial, cosmo,
               pt_model='folpsD', prior_basis='physical',
               width_EFT=12.5, width_SN0=2.0, width_SN2=5.0,
               b3_coev=True, A_full_status=False, damping='lor'):
    """Build coupled P(k) and B(k) theory models with nuisance priors.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.
    fiducial : :class:`cosmoprimo.Cosmology`
        Fiducial cosmology used for template evaluation.
    cosmo : :class:`Cosmoprimo`
        Cosmology calculator passed to the template.
    pt_model : str, optional
        Perturbation theory model for the power spectrum.  Supported values:
        ``'folpsD'`` (default), ``'EFT'``, ``'rept_velocileptors'``.
    prior_basis : str, optional
        Bias parametrisation basis.  Default ``'physical'``.
    width_EFT : float, optional
        Width of the Gaussian prior on EFT counter-terms.  Default 12.5.
    width_SN0 : float, optional
        Width of the Gaussian prior on the shot-noise monopole.  Default 2.0.
    width_SN2 : float, optional
        Width of the Gaussian prior on the shot-noise quadrupole.  Default 5.0.
    b3_coev : bool, optional
        Fix ``b3`` to its co-evolution value.  Default ``True``.
    A_full_status : bool, optional
        Use the full A-matrix in FOLPS.  Default ``False``.
    damping : str, optional
        Finger-of-god damping kernel.  One of ``'lor'``, ``'exp'``,
        ``'vdg'``.  Default ``'lor'``.

    Returns
    -------
    theories : dict
        ``{'ps': FOLPSv2TracerPowerSpectrumMultipoles,
           'bs': FOLPSv2TracerBispectrumMultipoles}``
        Both theory objects share the same template and have nuisance
        parameter priors applied.
    """
    z = TRACER_REDSHIFTS[tracer]
    sigma8_fid = fiducial.sigma8_z(z)
    template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)

    # ── PS theory ─────────────────────────────────────────────────────────
    if pt_model == 'rept_velocileptors':
        ps_theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(
            template=template,
            prior_basis=prior_basis,
        )
    else:
        ps_theory = FOLPSv2TracerPowerSpectrumMultipoles(
            template=template,
            prior_basis=prior_basis,
            A_full=A_full_status,
            b3_coev=b3_coev,
            damping=damping,
            sigma8_fid=sigma8_fid,
            h_fid=fiducial.h,
        )

    # ── BS theory ─────────────────────────────────────────────────────────
    bs_theory = FOLPSv2TracerBispectrumMultipoles(
        template=template,
        prior_basis=prior_basis,
        A_full=A_full_status,
        damping=damping,
        sigma8_fid=sigma8_fid,
        h_fid=fiducial.h,
    )

    theories = {'ps': ps_theory, 'bs': bs_theory}

    # Apply nuisance priors to whichever theory owns each parameter
    params = get_nuisance_priors(prior_basis, width_EFT, width_SN0, width_SN2,
                                 pt_model=pt_model, b3_coev=b3_coev,
                                 sigma8_fid=sigma8_fid)
    for name, p in params.items():
        for comp in ('ps', 'bs'):
            if name in theories[comp].params:
                theories[comp].params[name].update(p)

    return theories


# ─── Data loaders ────────────────────────────────────────────────────────────

def get_pk_data(tracer, region, k_min_p=0.02, k_max_p=0.20, n_data_mocks=25):
    """Average power spectrum multipoles over Abacus mocks.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.
    region : str
        Sky region, e.g. ``'SGC'``, ``'NGC'``, or ``'GCcomb'``.
    k_min_p : float, optional
        Minimum wavenumber [h/Mpc].  Default 0.02.
    k_max_p : float, optional
        Maximum wavenumber [h/Mpc].  Default 0.20.
    n_data_mocks : int, optional
        Number of Abacus mocks to average.  Default 25.

    Returns
    -------
    p0 : ndarray, shape (N_k,)
        Monopole data vector.
    p2 : ndarray, shape (N_k,)
        Quadrupole data vector.
    k_data : ndarray, shape (N_k,)
        k values [h/Mpc].
    pspectrum : lsstypes object
        Last-read spectrum object; needed for window matrix matching.
    """
    tracer_type, zrange = TRACER_TYPES[tracer]
    p0_list, p2_list = [], []
    for imock in range(n_data_mocks):
        fn_pk = get_stats_fn(
            stats_dir=_STATS_DIR,
            kind='mesh2_spectrum',
            version='abacus-2ndgen-complete',
            tracer=tracer_type,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            imock=imock,
        )
        pspectrum = types.read(fn_pk)
        pspectrum = pspectrum.select(k=slice(0, None, 5))
        pspectrum = pspectrum.select(k=(k_min_p, k_max_p))
        pspectrum = pspectrum.get(ells=[0, 2])
        p0_list.append(pspectrum.get(ells=0).value())
        p2_list.append(pspectrum.get(ells=2).value())

    p0 = np.mean(np.array(p0_list), axis=0)
    p2 = np.mean(np.array(p2_list), axis=0)
    k_data = pspectrum.get(ells=0).coords('k')
    return p0, p2, k_data, pspectrum


def get_bk_data(tracer, region, k_min_b=0.02, k_max_b0=0.12, k_max_b2=0.08,
                n_data_mocks=25):
    """Average bispectrum multipoles over Abacus mocks.

    Parameters
    ----------
    tracer : str
        Tracer label.
    region : str
        Sky region.
    k_min_b : float, optional
        Minimum wavenumber [h/Mpc] for both bispectrum multipoles.
        Default 0.02.
    k_max_b0 : float, optional
        Maximum wavenumber [h/Mpc] for the bispectrum monopole B000.
        Default 0.12.
    k_max_b2 : float, optional
        Maximum wavenumber [h/Mpc] for the bispectrum quadrupole B202.
        Default 0.08.
    n_data_mocks : int, optional
        Number of Abacus mocks to average.  Default 25.

    Returns
    -------
    b000 : ndarray
        Monopole bispectrum data vector.
    b202 : ndarray
        Quadrupole bispectrum data vector.
    kr_b0 : ndarray, shape (N_b0, 3)
        Triangle configurations for B000 (k1, k2, k3 columns).
    kr_b2 : ndarray, shape (N_b2, 3)
        Triangle configurations for B202 (k1, k2, k3 columns).
    """
    tracer_type, zrange = TRACER_TYPES[tracer]
    b000_list, b202_list = [], []
    for imock in range(n_data_mocks):
        fn_bk = get_stats_fn(
            stats_dir=_STATS_DIR,
            kind='mesh3_spectrum',
            version='abacus-2ndgen-complete',
            tracer=tracer_type,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            basis='sugiyama-diagonal',
            imock=imock,
        )
        bspectrum = types.read(fn_bk)
        bspectrum = bspectrum.select(k=(k_min_b, k_max_b0))
        bspectrum = bspectrum.get(ells=[(0, 0, 0), (2, 0, 2)])
        bspectrum = bspectrum.at(ells=(2, 0, 2)).select(k=(k_min_b, k_max_b2))
        b000_list.append(bspectrum.get(ells=(0, 0, 0)).value())
        b202_list.append(bspectrum.get(ells=(2, 0, 2)).value())

    b000 = np.mean(np.array(b000_list), axis=0)
    b202 = np.mean(np.array(b202_list), axis=0)
    kr_b0 = bspectrum.get(ells=(0, 0, 0)).coords('k')
    kr_b2 = bspectrum.get(ells=(2, 0, 2)).coords('k')
    return b000, b202, kr_b0, kr_b2, bspectrum


def get_joint_covariance(tracer, region,
                         k_min_p=0.02, k_max_p=0.20,
                         k_min_b=0.02, k_max_b0=0.12, k_max_b2=0.08,
                         n_cov_mocks=1000):
    """Build a Hartlap-corrected joint P(k)+B(k) covariance from Holi mocks.

    Parameters
    ----------
    tracer : str
        Tracer label.
    region : str
        Sky region.
    k_min_p : float, optional
        Minimum k [h/Mpc] for P(k).  Default 0.02.
    k_max_p : float, optional
        Maximum k [h/Mpc] for P(k).  Default 0.20.
    k_min_b : float, optional
        Minimum k [h/Mpc] for B(k).  Default 0.02.
    k_max_b0 : float, optional
        Maximum k [h/Mpc] for B000.  Default 0.12.
    k_max_b2 : float, optional
        Maximum k [h/Mpc] for B202.  Default 0.08.
    n_cov_mocks : int, optional
        Maximum number of Holi mocks to attempt.  Default 1000.

    Returns
    -------
    cov : ndarray
        Hartlap-corrected full joint covariance matrix for
        [P0, P2, B000, B202].
    cov_pk : ndarray
        P(k)-only upper-left block of the covariance.
    Nm : int
        Number of mocks successfully loaded.
    hartlap : float
        Hartlap correction factor applied to the covariance.
    """
    tracer_type, zrange = TRACER_TYPES[tracer]
    tracer_type_cov = 'ELG_LOPnotqso' if tracer_type == 'ELG_LOP' else tracer_type

    observables, missing, available = [], [], []
    for imock in range(n_cov_mocks):
        kw = dict(
            stats_dir=_STATS_DIR,
            version='holi-v1-altmtl',
            tracer=tracer_type_cov,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            imock=imock,
        )
        fn2 = get_stats_fn(kind='mesh2_spectrum', **kw)
        fn3 = get_stats_fn(kind='mesh3_spectrum', basis='sugiyama-diagonal', **kw)
        if not (fn2.exists() and fn3.exists()):
            missing.append(imock)
            continue
        available.append(imock)
        spectrum2 = types.read(fn2)
        spectrum3 = types.read(fn3)
        observables.append(ObservableTree(
            [spectrum2, spectrum3],
            observables=['spectrum2', 'spectrum3'],
        ))

    print(f"  Available mocks ({len(available)})")
    print(f"  Missing mocks   ({len(missing)})")

    if not observables:
        raise RuntimeError('No valid Holi mocks found for joint P+B covariance!')

    covariance = types.cov(observables)

    # Apply the same selections used for the data vectors
    observable = covariance.observable

    spectrum2 = observable.get(observables='spectrum2')
    spectrum2 = spectrum2.get(ells=[0, 2])
    spectrum2 = spectrum2.select(k=slice(0, None, 5))
    spectrum2 = spectrum2.select(k=(k_min_p, k_max_p))

    spectrum3 = observable.get(observables='spectrum3')
    spectrum3 = spectrum3.select(k=(k_min_b, k_max_b0))
    spectrum3 = spectrum3.at(ells=(2, 0, 2)).select(k=(k_min_b, k_max_b2))

    observable = observable.at(observables='spectrum2').match(spectrum2)
    observable = observable.at(observables='spectrum3').match(spectrum3)
    covariance = covariance.at.observable.match(observable)

    # Extract PS-only block and full matrix
    cov_pk = covariance.at.observable.get(observables='spectrum2').value()
    cov = covariance.value()

    Nm = len(available)
    Nd = cov.shape[0]
    hartlap = (Nm - Nd - 2) / (Nm - 1)
    print(f"  Total data vector length : {Nd}")
    print(f"  Hartlap factor           : {hartlap:.4f}")
    cov = cov / hartlap

    return cov, cov_pk, Nm, hartlap


# ─── Window matrices ─────────────────────────────────────────────────────────

def get_pk_window(tracer, region, spectrum):
    """Load and match the power spectrum convolution window matrix.

    Parameters
    ----------
    tracer : str
        Tracer label.
    region : str
        Sky region.
    spectrum : lsstypes object
        Spectrum object returned by :func:`get_pk_data`, used to match the
        window matrix rows to the observed k-bins and multipoles.

    Returns
    -------
    wmatnp : ndarray
        Window matrix array.
    k_window : ndarray
        Fine theory k grid [h/Mpc] for the window convolution.
    zeff : float
        Effective redshift stored in the window metadata.
    """
    tracer_type, zrange = TRACER_TYPES[tracer]
    window_fn = get_stats_fn(
        stats_dir=_STATS_DIR,
        kind='window_mesh2_spectrum',
        version='abacus-2ndgen-complete',
        tracer=tracer_type,
        zrange=zrange,
        region=region,
        weight='default-FKP',
        imock=0,
    )
    window = types.read(window_fn)
    window = window.at.observable.match(spectrum)
    window = window.at.theory.select(k=(0, 0.5))

    wmatnp = window.value()
    zeff = window.observable.get(ells=0).attrs['zeff']
    k_window = window.theory.get(ells=0).coords('k')
    return wmatnp, k_window, zeff


def get_bk_window(tracer, region, spectrum):
    """Load and match the bispectrum convolution window matrix via lsstypes.

    Reads the ``window_mesh3_spectrum`` (Sugiyama-diagonal basis) for the
    given tracer and sky region, matches its observable rows to *spectrum*,
    and trims the theory k grid to ``(0, 0.1)``.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.
    region : str
        Sky region, e.g. ``'SGC'``.
    spectrum : lsstypes object
        Bispectrum spectrum object returned by :func:`get_bk_data`, used to
        match the window matrix rows to the observed triangle bins and
        multipoles.

    Returns
    -------
    wmatnp : ndarray
        Window matrix array.
    k_window : ndarray
        Fine theory k grid [h/Mpc] for the window convolution.
    zeff : float
        Effective redshift stored in the window metadata.
    """
    tracer_type, zrange = TRACER_TYPES[tracer]
    window_fn = get_stats_fn(
        stats_dir=_STATS_DIR,
        kind='window_mesh3_spectrum',
        version='abacus-2ndgen-complete',
        tracer=tracer_type,
        zrange=zrange,
        region=region,
        weight='default-FKP',
        basis='sugiyama-diagonal',
        imock=0,
    )
    window = types.read(window_fn)
    window = window.at.observable.match(spectrum)
    window = window.at.theory.select(k=(0, 0.1))

    wmatnp = window.value()
    zeff = window.observable.get(ells=(0, 0, 0)).attrs['zeff']
    k_window = window.theory.get(ells=(0, 0, 0)).coords('k')
    return wmatnp, k_window, zeff


# ─── Observable ──────────────────────────────────────────────────────────────

def get_observable(tracer='LRG2', region='SGC',
                   k_min_p=0.02, k_max_p=0.20,
                   k_min_b=0.02, k_max_b0=0.12, k_max_b2=0.08,
                   n_data_mocks=25, n_cov_mocks=1000,
                   pt_model='folpsD', prior_basis='physical',
                   width_EFT=12.5, width_SN0=2.0, width_SN2=5.0,
                   b3_coev=True, A_full_status=False, damping='lor'):
    """Build coupled P(k) and B(k) observables for *tracer*.

    Orchestrates all data-loading, covariance, and window-matrix steps, then
    constructs two :class:`TracerPowerSpectrumMultipolesObservable` objects
    (one for P(k), one for B(k)) linked to shared theory calculators.

    Parameters
    ----------
    tracer : str, optional
        Tracer label.  Default ``'LRG2'``.
    region : str, optional
        Sky region.  Default ``'SGC'``.
    k_min_p : float, optional
        Minimum k [h/Mpc] for P(k).  Default 0.02.
    k_max_p : float, optional
        Maximum k [h/Mpc] for P(k).  Default 0.20.
    k_min_b : float, optional
        Minimum k [h/Mpc] for B(k).  Default 0.02.
    k_max_b0 : float, optional
        Maximum k [h/Mpc] for B000.  Default 0.12.
    k_max_b2 : float, optional
        Maximum k [h/Mpc] for B202.  Default 0.08.
    n_data_mocks : int, optional
        Number of Abacus mocks to average for the data vector.  Default 25.
    n_cov_mocks : int, optional
        Maximum number of Holi mocks used for the covariance.  Default 1000.
    pt_model : str, optional
        Perturbation theory model.  Default ``'folpsD'``.
    prior_basis : str, optional
        Bias parametrisation basis.  Default ``'physical'``.
    width_EFT : float, optional
        Width of EFT counter-term priors.  Default 12.5.
    width_SN0 : float, optional
        Width of shot-noise monopole prior.  Default 2.0.
    width_SN2 : float, optional
        Width of shot-noise quadrupole prior.  Default 5.0.
    b3_coev : bool, optional
        Fix ``b3`` to co-evolution value.  Default ``True``.
    A_full_status : bool, optional
        Use full A-matrix in FOLPS.  Default ``False``.
    damping : str, optional
        Finger-of-god damping kernel (``'lor'``, ``'exp'``, ``'vdg'``).
        Default ``'lor'``.

    Returns
    -------
    obs_dict : dict
        ``{'ps': TracerPowerSpectrumMultipolesObservable,
           'bs': TracerPowerSpectrumMultipolesObservable,
           'cov': ndarray,
           'theories': {'ps': ..., 'bs': ...}}``
        The ``'cov'`` entry is the full joint Hartlap-corrected covariance
        for [P0, P2, B000, B202], needed by :func:`get_likelihood`.
    """
    cosmo, fiducial = get_cosmology()

    theories = get_theory(
        tracer, fiducial, cosmo,
        pt_model=pt_model, prior_basis=prior_basis,
        width_EFT=width_EFT, width_SN0=width_SN0, width_SN2=width_SN2,
        b3_coev=b3_coev, A_full_status=A_full_status, damping=damping,
    )

    # ── P(k) data & window ────────────────────────────────────────────────
    p0, p2, k_data, pspectrum = get_pk_data(
        tracer, region, k_min_p, k_max_p, n_data_mocks,
    )
    wmatnp_pk, k_window_p, zeff = get_pk_window(tracer, region, pspectrum)
    print(f'  zeff = {zeff:.4f}')

    # ── B(k) data & window ────────────────────────────────────────────────
    b000, b202, kr_b0, kr_b2, bspectrum = get_bk_data(
        tracer, region, k_min_b, k_max_b0, k_max_b2, n_data_mocks,
    )
    wmatnp_bk, k_window_b, zeff_b = get_bk_window(tracer, region, spectrum=bspectrum)
    k_data_b = bspectrum.get(ells=(0, 0, 0)).coords('k')

    # ── Joint covariance ──────────────────────────────────────────────────
    cov, cov_pk, Nm, hartlap = get_joint_covariance(
        tracer, region,
        k_min_p, k_max_p,
        k_min_b, k_max_b0, k_max_b2,
        n_cov_mocks,
    )

    n_pk = p0.size + p2.size
    n_bk = b000.size + b202.size
    assert cov.shape[0] == n_pk + n_bk, (
        f'Covariance size {cov.shape[0]} does not match '
        f'total data vector length {n_pk + n_bk}'
    )

    # ── Build observables ─────────────────────────────────────────────────
    ps_obs = TracerPowerSpectrumMultipolesObservable(
        data=np.concatenate([p0, p2]),
        covariance=cov_pk,
        theory=theories['ps'],
        kin=k_window_p,
        ellsin=[0, 2, 4],
        ells=(0, 2),
        k=k_data,
        wmatrix=wmatnp_pk,
    )

    bs_obs = TracerPowerSpectrumMultipolesObservable(
        data=np.concatenate([b000, b202]),
        covariance=cov[n_pk:, n_pk:],
        theory=theories['bs'],
        ells=(0, 2),
        k=[kr_b0,kr_b2], 
        # kin=k_window_b,
        # ellsin=[(0, 0, 0), (2, 0, 2)],
        # ells=[(0, 0, 0), (2, 0, 2)],
        # k=k_data_b,
        # wmatrix=wmatnp_bk,
    )

    return {'ps': ps_obs, 'bs': bs_obs, 'cov': cov, 'theories': theories}


# ─── Emulator ────────────────────────────────────────────────────────────────

def set_emulator(observables, emulator_dir='./emulators', order=4):
    """Fit or load Taylor emulators for the PS and BS theory calculators.

    For each tracer and each component (``'ps'``, ``'bs'``), the emulator is
    saved to / loaded from ``{emulator_dir}/{tracer}_{comp}_emulator.npy``.

    Parameters
    ----------
    observables : dict
        Mapping ``{tracer: obs_dict}`` as returned by :func:`get_observable`.
    emulator_dir : str or Path, optional
        Directory for emulator files.  Default ``'./emulators'``.
    order : int, optional
        Taylor expansion order for the emulator.  Default 4.

    Returns
    -------
    observables : dict
        The same dictionary with each theory's PT calculator replaced by an
        emulated version.
    """
    emulator_dir = Path(emulator_dir)
    for tracer, obs_dict in observables.items():
        theories = obs_dict['theories']
        for comp in ('ps', 'bs'):
            obs = obs_dict[comp]
            filename = emulator_dir / f'{tracer}_{comp}_emulator.npy'
            filename.parent.mkdir(parents=True, exist_ok=True)

            if filename.exists():
                print(f'Loading {comp.upper()} emulator for {tracer}')
                emulator = EmulatedCalculator.load(filename)
                theories[comp].init.update(pt=emulator)
            else:
                print(f'Fitting {comp.upper()} emulator for {tracer}')
                # Both PS and BS observables now carry a window matrix, so
                # the PT calculator is always accessed via obs.wmatrix.theory.
                theory_calc = obs.wmatrix.theory
                emulator = Emulator(
                    theory_calc.pt,
                    engine=TaylorEmulatorEngine(method='finite', order=order),
                )
                emulator.set_samples()
                emulator.fit()
                emulated_pt = emulator.to_calculator()
                emulated_pt.save(filename)
                theories[comp].init.update(pt=emulated_pt)

    return observables


# ─── Analytic marginalisation ────────────────────────────────────────────────

def set_analytic_marginalization(observables, prior_basis):
    """Mark PS EFT counter-term and shot-noise parameters for analytic marginalisation.

    Only the power-spectrum counter-terms (``alpha0``, ``alpha2``, ``alpha4``,
    ``sn0``, ``sn2``) are marginalised analytically.  Bispectrum stochastic
    parameters (``c1``, ``c2``, ``Pshot``, ``Bshot``) are sampled explicitly.

    Parameters
    ----------
    observables : dict
        Mapping ``{tracer: obs_dict}`` as returned by :func:`get_observable`.
    prior_basis : str
        Bias parametrisation basis.  ``'physical'`` or
        ``'physical_prior_doc'`` marginalises ``alpha0p``, ``alpha2p``,
        ``alpha4p``, ``sn0p``, ``sn2p``; any other value marginalises their
        un-suffixed counterparts.

    Returns
    -------
    observables : dict
        The same dictionary with marginalisation flags set in-place.
    """
    for obs_dict in observables.values():
        if prior_basis in ('physical', 'physical_prior_doc'):
            params_list = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
        else:
            params_list = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']
        for param in params_list:
            obs_dict['ps'].wmatrix.theory.params[param].update(derived='.marg')
    return observables


# ─── Likelihood ──────────────────────────────────────────────────────────────

def get_likelihood(observables):
    """Build a joint P(k)+B(k) Gaussian likelihood for each tracer, summed
    over all tracers.

    Each tracer contributes a single :class:`ObservablesGaussianLikelihood`
    that wraps both the PS and BS observables with the full joint covariance.

    Parameters
    ----------
    observables : dict
        Mapping ``{tracer: obs_dict}`` as returned by :func:`get_observable`.

    Returns
    -------
    :class:`SumLikelihood`
        Joint log-likelihood over all tracers.
    """
    likelihoods = []
    for obs_dict in observables.values():
        likelihoods.append(
            ObservablesGaussianLikelihood(
                observables=[obs_dict['ps'], obs_dict['bs']],
                covariance=obs_dict['cov'],
            )
        )
    return SumLikelihood(likelihoods)


# ─── MCMC ────────────────────────────────────────────────────────────────────

def run_mcmc(likelihood, chain_name, GR_criteria=0.3):
    """Run an MCMC chain using :class:`EmceeSampler`.

    Parameters
    ----------
    likelihood : :class:`SumLikelihood`
        The joint likelihood to sample.
    chain_name : str
        Path prefix used by the sampler when saving chain files.
    GR_criteria : float, optional
        Gelman–Rubin convergence threshold on the maximum eigenvalue of the
        GR statistic.  Sampling stops when this criterion is met.
        Default 0.3.
    """
    sampler = EmceeSampler(likelihood, save_fn=chain_name)
    sampler.run(check={'max_eigen_gr': GR_criteria})


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Fit P(k)+B(k) multipoles for DESI cutsky mocks.'
    )
    parser.add_argument('--tracers', nargs='+', type=str, default=['LRG2'],
                        help='Tracer labels (default: LRG2).')
    parser.add_argument('--region', type=str, default='GCcomb',
                        help='Sky region (default: GCcomb).')
    parser.add_argument('--emulator_dir', type=str, default='./emulators',
                        help='Directory to save/load emulators.')
    parser.add_argument('--pt_model', type=str, default='folpsD',
                        help='PT model: folpsD, EFT, rept_velocileptors.')
    parser.add_argument('--prior_basis', type=str, default='standard',
                        help='Bias parametrisation: physical, physical_prior_doc, standard.')
    parser.add_argument('--k_max_p', type=float, default=0.20,
                        help='Max k [h/Mpc] for P(k) (default: 0.20).')
    parser.add_argument('--k_max_b0', type=float, default=0.12,
                        help='Max k [h/Mpc] for B000 (default: 0.12).')
    parser.add_argument('--k_max_b2', type=float, default=0.08,
                        help='Max k [h/Mpc] for B202 (default: 0.08).')
    parser.add_argument('--GR_criteria', type=float, default=0.3,
                        help='Gelman-Rubin convergence threshold (default: 0.3).')
    args = parser.parse_args()

    setup_logging()

    observables = {}
    for tracer, region in itertools.product(args.tracers, [args.region]):
        print(f'\n{"="*60}')
        print(f'Building observable: tracer={tracer}  region={region}')
        print(f'{"="*60}')
        observables[tracer] = get_observable(
            tracer=tracer,
            region=region,
            k_max_p=args.k_max_p,
            k_max_b0=args.k_max_b0,
            k_max_b2=args.k_max_b2,
            pt_model=args.pt_model,
            prior_basis=args.prior_basis,
        )

    observables = set_emulator(observables, emulator_dir=args.emulator_dir)
    observables = set_analytic_marginalization(observables, prior_basis=args.prior_basis)

    likelihood = get_likelihood(observables)

    tracer_str = '_'.join(args.tracers)
    chain_name = f'./chain_pbk_{tracer_str}_{args.region}_kp{args.k_max_p:.2f}_kb0{args.k_max_b0:.2f}_kb2{args.k_max_b2:.2f}'
    run_mcmc(likelihood, chain_name, GR_criteria=args.GR_criteria)
