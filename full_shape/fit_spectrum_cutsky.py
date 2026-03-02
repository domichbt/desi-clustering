import os
os.environ["FOLPS_BACKEND"] = "jax" 
import itertools

import numpy as np
import matplotlib.pyplot as plt
from desilike.theories.galaxy_clustering import REPTVelocileptorsTracerPowerSpectrumMultipoles,  DirectPowerSpectrumTemplate 
from desilike.theories.galaxy_clustering.full_shape import FOLPSv2TracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable 
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine
from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import DESI
from desilike import setup_logging
from desilike.samplers import EmceeSampler
from desilike import ParameterCollection
import numpy as np
from pathlib import Path

from clustering_statistics.tools import get_stats_fn
import lsstypes as types


TRACER_TYPES: dict = {
    'LRG1': ('LRG',     (0.4, 0.6)),
    'LRG2': ('LRG',     (0.6, 0.8)),
    'LRG3': ('LRG',     (0.8, 1.1)),
    'QSO':  ('QSO',     (0.8, 2.1)),
    'ELG':  ('ELG_LOP', (1.1, 1.6)),
}

TRACER_REDSHIFTS = {
    'BGS': 0.295,
    'LRG1': 0.5094,
    'LRG2': 0.7054,
    'LRG3': 0.9264,
    'ELG': 1.3442,
    'QSO': 1.4864
}

TRACER_SIGMA8 = {
    'BGS': 0.69376997,
    'LRG1':0.62404056,
    'LRG2': 0.53930063,
    'LRG3': 0.50823223,
    'ELG': 0.43197292,
    'QSO': 0.41825647
}

_STATS_DIR = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')


def get_nuisance_priors(prior_basis, width_EFT, width_SN0, width_SN2, pt_model='folpsD', b3_coev=True, sigma8_fid=None):
    """Build a :class:`ParameterCollection` of nuisance parameter priors.

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
        damping parameter is fixed.  Default ``'folpsD'``.
    b3_coev=True : bool, optional
        If ``True``, the third-order bias ``b3`` is fixed to its co-evolution
        value and not sampled.  Default ``True``.
    sigma8_fid : float, optional
        Fiducial :math:`\sigma_8` at the tracer effective redshift, used to
        set the prior centre for ``bsp`` in the physical basis.

    Returns
    -------
    params : :class:`ParameterCollection`
        Collection of nuisance parameters with priors attached.
    """
    params = ParameterCollection()

    if prior_basis in ('physical', 'physical_prior_doc'):
        # Shared params
        params['b1p']= {'prior': {'dist': 'uniform', 'limits':[0.1,4]}}
        params['b2p']= {'prior': {'dist': 'norm', 'loc': 0, 'scale': 5}}
        params['bsp'] = {'prior': {'dist': 'norm', 'loc': -2/7*(sigma8_fid)**2, 'scale': 5}}
        if b3_coev:
            params['b3p'] = {'fixed':True}
            # params['bsp'] = {'fixed':True}
        else:
            params['b3p'] = {'prior': {'dist': 'norm', 'loc': 23/42*(sigma8_fid)**4, 'scale': 1*sigma8_fid**4},'fixed':False}#TBD
        
        # PS-only
        params['alpha0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2p'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        #Maximal Freedom
        # params['alpha0p'] = {'prior': {'dist':'uniform','limits': [-100, 100]}}
        # params['alpha2p'] = {'prior': {'dist':'uniform','limits': [-100, 100]}}
        # params['alpha4p'] = {'prior': {'dist':'uniform','limits': [-100, 100]}}
        # params['sn0p'] = {'prior': {'dist':'uniform','limits': [-10, 10]}}
        # params['sn2p'] = {'prior': {'dist':'uniform','limits': [-10, 10]}}
        if pt_model=='EFT': 
            params['X_FoG_pp'] = {'fixed':True}
        else: 
            params['X_FoG_pp'] = {'prior': {'dist':'uniform','limits': [0, 10]}} 
        # BS-only → if physical, no c1,c2,Pshot,Bshot,X_FoG_b
        # params['c1'] = {'prior': {'dist':'uniform','limits': [-2000, 2000]}}
        # params['c2'] = {'prior': {'dist':'uniform','limits': [-2000, 2000]}}
        # params['Pshot'] = {'prior': {'dist':'uniform','limits': [-50000, 50000]}}
        # params['Bshot'] = {'prior': {'dist':'uniform','limits': [-50000, 50000]}}
        # params['X_FoG_b'] = {'prior': {'dist':'uniform','limits': [0, 15]}}

    else:
        # Shared params
        params['b1'] = {'prior': {'dist':'uniform','limits': [1e-5, 10]}}
        params['b2'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        params['bs'] = {'prior': {'dist':'uniform','limits': [-50, 50]}}
        if b3_coev:
            params['b3'] = {'fixed':True}
        else:
            params['b3'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': 1}}#TBD
        # params['b3'] = {'fixed':True}

        # PS-only
        params['alpha0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['alpha4'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_EFT}}
        params['sn0'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN0}}
        params['sn2'] = {'prior': {'dist': 'norm', 'loc': 0, 'scale': width_SN2}}
        # params['alpha0'] = {'value':20,'fixed':True}
        # params['alpha2'] = {'value':-58.8,'fixed':True}
        # params['alpha4'] = {'value':0.0,'fixed':True}
        # params['sn0'] = {'value':-0.073,'fixed':True}
        # params['sn2'] = {'value':-6.38,'fixed':True}
        if pt_model=='EFT':
            params['X_FoG_p'] = {'fixed':True}
            params['X_FoG_b'] = {'fixed':True}
        else: 
            params['X_FoG_p'] = {'prior': {'dist':'uniform','limits': [0, 10]}} 
            params['X_FoG_b'] = {'prior': {'dist':'uniform','limits': [0, 15]}}
        # params['X_FoG_p'] = {'fixed':True}  # fixed in your snippet

        # BS-only
        params['c1'] = {'prior': {'dist':'uniform','limits': [-2000, 2000]}}
        params['c2'] = {'prior': {'dist':'uniform','limits': [-2000, 2000]}}
        params['Pshot'] = {'prior': {'dist':'uniform','limits': [-50000, 50000]}}
        params['Bshot'] = {'prior': {'dist':'uniform','limits': [-50000, 50000]}}
        
    return params


def get_cosmology():
    """Initialise a :class:`Cosmoprimo` cosmology and a DESI fiducial cosmology.

    Sets up a CLASS-based cosmology calculator with free parameters
    ``h``, ``omega_cdm``, ``omega_b``, ``logA`` (uniform priors) and
    ``n_s``, ``tau_reio`` fixed to their fiducial values.  A Gaussian prior
    is placed on ``omega_b`` from simulations.

    Returns
    -------
    cosmo : :class:`Cosmoprimo`
        Cosmology calculator with priors configured for sampling.
    fiducial : :class:`cosmoprimo.Cosmology`
        DESI fiducial cosmology used for template evaluation.
    """
    #Define a cosmology to get sigma_8 and Omega_m
    cosmo = Cosmoprimo(engine='class')
    cosmo.init.params['H0'] = dict(derived=True)
    cosmo.init.params['Omega_m'] = dict(derived=True)
    cosmo.init.params['sigma8_m'] = dict(derived=True) 
    fiducial = DESI() #fiducial cosmology

    #Update cosmo priors
    for param in ['n_s', 'h','omega_cdm', 'omega_b', 'logA', 'tau_reio']:
        cosmo.params[param].update(fixed = False)
        if param == 'tau_reio':
            cosmo.params[param].update(fixed = True)
        if param == 'n_s':
                cosmo.params[param].update(fixed = True)
                # cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042})
        if param == 'omega_b':
                # cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02218, 'scale': 0.00055})
                cosmo.params[param].update(prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})  #From simulations
            # ,ref={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00015}
        if param == 'h':
                cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.5,0.9]})
            # ,ref={'dist': 'norm', 'loc': 0.6736, 'scale': 0.005}
        if param == 'omega_cdm':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.05, 0.2]})
            # ,ref={'dist': 'norm', 'loc': 0.12, 'scale': 0.0012}
        #if param == 'm_ncdm':
            #cosmo.params[param].update(prior = {'dist':'uniform','limits': [0.0, 5]})
        if param == 'logA':
            cosmo.params[param].update(prior = {'dist':'uniform','limits': [2.0, 4.0]})
            # ,ref={'dist': 'norm', 'loc': 3.036394, 'scale': 0.014}

    return cosmo, fiducial


def get_theory(tracer, fiducial, cosmo, pt_model='folpsD', prior_basis='physical', width_EFT=10, width_SN0=1, width_SN2=1, b3_coev=True, A_full_status=False, damping=False):
    """Build a power spectrum theory model with nuisance priors for *tracer*.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.  Must be a key in ``TRACER_REDSHIFTS``.
    fiducial : :class:`cosmoprimo.Cosmology`
        Fiducial cosmology used to evaluate the power spectrum template.
    cosmo : :class:`Cosmoprimo`
        Cosmology calculator passed to the template.
    pt_model : str, optional
        Perturbation theory model.  Supported values:
        ``'folpsD'`` (default), ``'TNS'``, ``'rept_velocileptors'``.
    prior_basis : str, optional
        Bias parametrisation basis passed to the theory model and to
        :func:`get_nuisance_priors`.  Default ``'physical'``.
    width_EFT : float, optional
        Width of the Gaussian prior on EFT counter-terms. Default 10.
    width_SN0 : float, optional
        Width of the Gaussian prior on the shot-noise monopole. Default 1.
    width_SN2 : float, optional
        Width of the Gaussian prior on the shot-noise quadrupole. Default 1.
    b3_coev : bool, optional
        Fix ``b3`` to its co-evolution value. Default ``True``.
    A_full_status : bool, optional
        Use the full A-matrix in FOLPS. Default ``False``.
    damping : bool, optional
        Enable damping terms in FOLPS. Default ``False``.

    Returns
    -------
    theory : :class:`FOLPSv2TracerPowerSpectrumMultipoles` or equivalent
        Theory calculator with nuisance parameter priors attached.
    """
    z = TRACER_REDSHIFTS[tracer]
    sigma8_fid = fiducial.sigma8_z(z)
    template = DirectPowerSpectrumTemplate(fiducial=fiducial, cosmo=cosmo, z=z)

    if pt_model == "rept_velocileptors":
        theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(
            template=template,
            prior_basis=prior_basis
        )
    else:
        if pt_model == 'TNS':
            theory = FOLPSv2TracerPowerSpectrumMultipoles(
                template=template,
                prior_basis=prior_basis,
                A_full=A_full_status,
                b3_coev=b3_coev,
                damping=damping,
                remove_DeltaP=True
            )
        else:
            theory = FOLPSv2TracerPowerSpectrumMultipoles(
                template=template,
                prior_basis=prior_basis,
                A_full=A_full_status,
                b3_coev=b3_coev,
                damping=damping,
                sigma8_fid=sigma8_fid,
                h_fid=fiducial.h
            )

    params = get_nuisance_priors(prior_basis, width_EFT, width_SN0, width_SN2, pt_model=pt_model,b3_coev=b3_coev,sigma8_fid=sigma8_fid)
    for name, p in params.items():
        if name in theory.params:
            theory.params[name].update(p)
    return theory


def get_data(tracer, region, k_min_p=0.02, k_max_p=0.20, n_data_mocks=25):
    """Average power spectrum multipoles over Abacus mocks.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.
    region : str
        Sky region, e.g. ``'SGC'``, ``'NGC'``, or ``'GCcomb'``.
    k_min_p : float, optional
        Minimum wavenumber [h/Mpc]. Default 0.02.
    k_max_p : float, optional
        Maximum wavenumber [h/Mpc]. Default 0.20.
    n_data_mocks : int, optional
        Number of Abacus mocks to average. Default 25.

    Returns
    -------
    p0 : ndarray
        Monopole data vector, shape (N_k,).
    p2 : ndarray
        Quadrupole data vector, shape (N_k,).
    k_data : ndarray
        k values [h/Mpc], shape (N_k,).
    pspectrum : lsstypes object
        Last-read spectrum object (needed to match the window matrix).
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


def get_covariance(tracer, region, k_min_p=0.02, k_max_p=0.20, n_cov_mocks=1000):
    """Build a Hartlap-corrected power spectrum covariance from Holi mocks.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.
    region : str
        Sky region.
    k_min_p : float, optional
        Minimum wavenumber [h/Mpc]. Default 0.02.
    k_max_p : float, optional
        Maximum wavenumber [h/Mpc]. Default 0.20.
    n_cov_mocks : int, optional
        Maximum number of Holi mocks to try. Default 1000.

    Returns
    -------
    cov_pk : ndarray
        Hartlap-corrected covariance for [P0, P2].
    Nm : int
        Number of mocks found and used.
    hartlap : float
        Hartlap correction factor.
    """
    tracer_type, zrange = TRACER_TYPES[tracer]
    tracer_type_cov = 'ELG_LOPnotqso' if tracer_type == 'ELG_LOP' else tracer_type

    observables, missing, available = [], [], []
    for imock in range(n_cov_mocks):
        fn2 = get_stats_fn(
            stats_dir=_STATS_DIR,
            kind='mesh2_spectrum',
            version='holi-v1-altmtl',
            tracer=tracer_type_cov,
            zrange=zrange,
            region=region,
            weight='default-FKP',
            imock=imock,
        )
        if not fn2.exists():
            missing.append(imock)
            continue
        available.append(imock)
        spectrum2 = types.read(fn2)
        observables.append(types.ObservableTree([spectrum2], observables=['spectrum2']))

    print(f"  Available mocks ({len(available)})")
    print(f"  Missing mocks   ({len(missing)})")

    if not observables:
        raise RuntimeError('No valid Holi mocks found for covariance!')

    covariance = types.cov(observables)

    observable = covariance.observable
    spectrum2 = observable.get(observables='spectrum2')
    spectrum2 = spectrum2.get(ells=[0, 2])
    spectrum2 = spectrum2.select(k=slice(0, None, 5))
    spectrum2 = spectrum2.select(k=(k_min_p, k_max_p))
    observable = observable.at(observables='spectrum2').match(spectrum2)
    covariance = covariance.at.observable.match(observable)

    cov_pk = covariance.value()
    Nm = len(available)
    Nd = cov_pk.shape[0]
    hartlap = (Nm - Nd - 2) / (Nm - 1)
    print(f"  Hartlap factor: {hartlap:.4f}")
    cov_pk = cov_pk / hartlap
    return cov_pk, Nm, hartlap


def get_window(tracer, region, spectrum):
    """Load and match the power spectrum window matrix.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.
    region : str
        Sky region.
    spectrum : lsstypes object
        Spectrum object (from :func:`get_data`) used to match the window
        matrix rows to the observed k-bins and multipoles.

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


def get_observable(tracer='LRG2', region='SGC', z_ev=0.8, k_max=0.301, k_max_b0=None, k_max_b2=None, P4=False):
    """Build a :class:`TracerPowerSpectrumMultipolesObservable` for *tracer*.

    Calls :func:`get_data`, :func:`get_covariance`, :func:`get_window`, and
    :func:`get_theory` to assemble the data vector, covariance, window matrix,
    and theory model, then returns a desilike observable ready for likelihood
    evaluation or emulation.

    Parameters
    ----------
    tracer : str, optional
        Tracer label.  Default ``'LRG2'``.
    region : str, optional
        Sky region.  Default ``'SGC'``.
    z_ev : float, optional
        Evaluation redshift (currently unused; kept for API compatibility).
        Default 0.8.
    k_max : float, optional
        Maximum wavenumber [h/Mpc] for the power spectrum data vector.
        Default 0.301.
    k_max_b0 : float, optional
        Maximum wavenumber [h/Mpc] for the bispectrum monopole
        (currently unused).  Defaults to 0.12 if ``None``.
    k_max_b2 : float, optional
        Maximum wavenumber [h/Mpc] for the bispectrum quadrupole
        (currently unused).  Defaults to 0.08 if ``None``.
    P4 : bool, optional
        Include the hexadecapole (currently unused).  Default ``False``.

    Returns
    -------
    :class:`TracerPowerSpectrumMultipolesObservable`
        Configured observable including data, covariance, window matrix,
        and theory model.
    """
    # To avoid getting an error
    if k_max_b0==None:
        k_max_b0 = 0.12
    if k_max_b2==None:
        k_max_b2=0.08

    cosmo, fiducial = get_cosmology()

    p0, p2, k, spectrum = get_data(tracer, region, k_max_p=k_max)
    data = np.concatenate([p0, p2])
    covariance = get_covariance(tracer, region, k_max_p=k_max)[0]  # get covariance
    wmatrix, k_window, zeff = get_window(tracer, region, spectrum=spectrum)  # get window matrix and matching k grid

    theory = get_theory(
        tracer,
        fiducial,
        cosmo,
        pt_model=args.spectrum_theory,
        prior_basis=args.prior_basis,
        width_EFT=10,
        width_SN0=1,
        width_SN2=1, b3_coev=True, A_full_status=False, damping=False)

    return TracerPowerSpectrumMultipolesObservable(
        data=data,
        covariance=covariance,
        theory=theory,
        kin=k_window,
        ellsin=[0,2,4],
        ells=(0,2),
        k=k, wmatrix=wmatrix
    )

def set_emulator(observables, kr_max=0.3, A_full_status=False):
    """Fit or load a Taylor emulator for the PT theory in each observable.

    For each tracer the emulator is saved to / loaded from
    ``{args.emulator_dir}/{tracer}_emulator.npy``.  If the file already
    exists it is loaded directly; otherwise a 2nd-order Taylor emulator
    is fitted and saved.

    Parameters
    ----------
    observables : dict
        Mapping ``{tracer: TracerPowerSpectrumMultipolesObservable}`` as
        returned by :func:`get_observable`.
    kr_max : float, optional
        Maximum wavenumber passed to the emulator (currently informational
        only).  Default 0.3.
    A_full_status : bool, optional
        Passed through for reference (currently informational only).
        Default ``False``.

    Returns
    -------
    observables : dict
        The same dictionary with each observable's PT theory replaced by
        an emulated calculator.
    """
    for tracer, observable in observables.items():

        filename = Path(args.emulator_dir) / f'{tracer}_emulator.npy'
        
        if os.path.exists(filename):
            print(f"Emulator for tracer {tracer} already exists, loading it now")
            emulator = EmulatedCalculator.load(filename)
            observable.wmatrix.theory.init.update(pt=emulator)
        else:
            print(f"Fitting emulator for tracer {tracer}")
            theory = observable.wmatrix.theory
            emulator = Emulator(
                theory.pt,
                engine=TaylorEmulatorEngine(method='finite', order=2)
            )
            emulator.set_samples()
            emulator.fit()
            emulated_pt = emulator.to_calculator()
            emulated_pt.save(filename)
            observable.wmatrix.theory.init.update(pt=emulated_pt)
    return observables

def set_analytic_marginalization(observables, prior_basis):
    """Mark EFT counter-term and shot-noise parameters for analytic marginalisation.

    Sets the ``derived`` attribute of each counter-term / stochastic parameter
    to ``'.marg'`` so that desilike integrates them out analytically during
    sampling.

    Parameters
    ----------
    observables : dict
        Mapping ``{tracer: TracerPowerSpectrumMultipolesObservable}``.
    prior_basis : str
        Bias parametrisation basis.  ``'physical'`` or
        ``'physical_prior_doc'`` marginalises ``alpha0p``, ``alpha2p``,
        ``alpha4p``, ``sn0p``, ``sn2p``; any other value marginalises their
        un-suffixed counterparts.

    Returns
    -------
    observables : dict
        The same dictionary with marginalisation configured in-place.
    """
    for observable in observables.values():
        if prior_basis in ('physical', 'physical_prior_doc'):
            params_list = ['alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']
        else:
             params_list = ['alpha0', 'alpha2', 'alpha4', 'sn0', 'sn2']
        for param in params_list:    
            observable.wmatrix.theory.params[param].update(derived = '.marg')
    return observables


def get_likelihood(observables):
    """Build a joint Gaussian likelihood from a collection of observables.

    Wraps each observable in an :class:`ObservablesGaussianLikelihood` and
    combines them into a :class:`SumLikelihood`.

    Parameters
    ----------
    observables : dict
        Mapping ``{tracer: TracerPowerSpectrumMultipolesObservable}``.

    Returns
    -------
    :class:`SumLikelihood`
        Joint log-likelihood over all tracers.
    """
    likelihoods = []
    for observable in observables.values():
            likelihoods.append(ObservablesGaussianLikelihood(observable))
    return SumLikelihood(likelihoods)
        

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
        GR statistic.  Sampling stops when this is reached.  Default 0.3.
    """
    sampler = EmceeSampler(likelihood, save_fn=chain_name)
    sampler.run(check={'max_eigen_gr': GR_criteria})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracers', nargs='+', type=str, default=['LRG1'], help='Tracers')
    parser.add_argument('--region', type=str, default='GCcomb', help='Sky region to include.')
    parser.add_argument('--emulator_dir', type=str, default='./emulators', help='Directory to save/load emulators.')
    parser.add_argument('--spectrum_theory', type=str, default='rept_velocileptors', help='Perturbation theory model for the power spectrum.')
    parser.add_argument('--prior_basis', type=str, default='physical', help='Bias parametrisation basis.')
    args = parser.parse_args()

    setup_logging()

    observables = {}
    for tracer, region in itertools.product(args.tracers, [args.region]):
        obs = get_observable(tracer=tracer, region=region)
        observables[tracer] = obs

    observables = set_emulator(observables, kr_max=0.3, A_full_status=False)
    observables = set_analytic_marginalization(observables, prior_basis=args.prior_basis)

    likelihood = get_likelihood(observables)

    chain_name = './test'
    run_mcmc(likelihood, chain_name, GR_criteria=0.3)