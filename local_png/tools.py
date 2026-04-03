import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('PNG fitting tools')


def read_data(stats_dir='.', tracer='LRG', zrange=(0.4, 1.1), weight_type='default-fkp-oqe', region='GCcomb'):
    """ 
    Read the data from the clustering statistics output. This is a wrapper of clustering_statistics.tools.get_stats_fn.
    """
    import lsstypes
    from clustering_statistics import tools
    import functools 
    
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, tracer=tracer, zrange=zrange, weight=weight_type, region=region)

    # Read the data:
    pk = lsstypes.read(get_stats_fn(kind='mesh2_spectrum'))
    window = lsstypes.read(get_stats_fn(kind='window_mesh2_spectrum'))
    cov = lsstypes.read(get_stats_fn(kind='covariance_mesh2_spectrum'))

    return pk, window, cov


def rebin_data(pk, window, cov, tracer='LRG', kmin=1e-3, kmax=0.08, kpivot=2e-2, nrebin=2, use_ell2=True):
    """ 
    Rebin the data with k > kpivot by a factor nrebin. The quadrupole is rebinned again by a factor nrebin for the full range.
    Then, select data in the k range [kmin, kmax]. If use_ell2 is False, we only keep the monopole.
    Finally, we match the size of the window and covariance to the size of the power spectrum.
    Return the rebinned power spectrum, window and covariance.
    """
    tracers = tuple(tracer.split('x')) 
    if len(tracers) == 1: tracers *= 2

    # Let's rebin the power spectrum : 
    # print(f'Original k shape (ell=0): {pk.get(0).k.shape[0]}')
    pk = pk.map(lambda pole: pole.at(k=(kpivot, 1.)).select(k=slice(0, None, nrebin)))
    pk = pk.at(2).select(k=slice(0, None, nrebin))  # Rebin the quadrupole again but for the full range.
    # kpivot, nrebin = 4e-2, 2
    # pk = pk.map(lambda pole: pole.at(k=(kpivot, 1.)).select(k=slice(0, None, nrebin)))
    pk = pk.select(k=(kmin, kmax))
    if not use_ell2: pk = pk.get(ells=[0])

    # Match the size of wmatrix and covariance: 
    window = window.at.observable.match(pk)
    cov = cov.at.observable.at(observables='spectrum2', tracers=tracers).match(pk)
    
    logger.info(f'After rebinning and k range selection: {pk.get(0).k.shape[0]} and {pk.get(2).k.shape[0] if use_ell2 else "Not used"} data points.')

    return pk, window, cov


def plot_observables(observables):
    """ 
    Plot the observables (power spectrum multipoles) with their theory predictions and residuals. 
    """
    fig, axs = plt.subplots(2, 2,  figsize=(6, 4), sharex=True, sharey=False, gridspec_kw={'height_ratios': (3, 1)}, squeeze=True)
    fig.subplots_adjust(hspace=0.1)

    for tracer in observables.keys():
        for obs in observables[tracer]:
            j = 1 if 'ell2' in obs.name else 0

            wtheory = obs.data.clone(value=obs.flattheory)
           
            data_pole = obs.data.get()
            wtheory_pole = wtheory.get()
            x = data_pole.coords('k')
            std = obs.covariance.at.observable.get().std()

            scale = 1.
            axs[0, j].errorbar(x, scale * data_pole.value(), yerr=scale * std, linestyle='none', marker='o', markersize=4, label=rf'{tracer}')
            axs[0, j].loglog(x, scale * wtheory_pole.value(), ls='-', c='k')

            axs[1, j].plot(x, (data_pole.value() - wtheory_pole.value()) / std)
            axs[1, j].set_ylim(-4, 4)
            for offset in [-2., 2.]: axs[1, j].axhline(offset, color='k', linestyle='--')

    axs[0, 0].set_ylim(1e4, 8e4)
    axs[0, 1].set_ylim(2e3, 5e4)

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 0].set_title(r'$\ell = 0$', fontsize=10)
    axs[0, 1].set_title(r'$\ell = 2$', fontsize=10)
    axs[0, 0].set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{3}$]')
    axs[1, 0].set_ylabel(r'$\Delta P_{\ell} / \sigma (P_{\ell})$')
    axs[1, 0].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    axs[1, 1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    plt.tight_layout()
    plt.show()


def run_profiler(likelihood, fn_output=None):
    """ 
    Run the iminuit profiler on the likelihood, the results are saved in a text file if output_name is provided. fn_output should be a .txt file. 
    """
    from desilike.profilers import MinuitProfiler

    profiler = MinuitProfiler(likelihood, seed=7)
    profiler.maximize(niterations=20)
    logger.info(f'\n{profiler.profiles.to_stats(tablefmt="pretty")}')

    if fn_output is not None:
        _ = profiler.profiles.to_stats(fn=fn_output)
        np.savetxt(fn_output.replace('.txt', '_list.txt'), profiler.profiles.to_stats(tablefmt='list')[0], fmt='%s')


    """"""
    
    """ 

    nchains
    """

    """_summary_

    Returns:
        _type_: _description_
    """

def run_mcmc(likelihood, fn_output='tmp/mcmc_output_*.npy', extend_chains=False, nchains=1, max_iterations=1e5, check_every=1000):
    """Run the MCMC sampler on the likelihood, the results are saved in a text file if fn_output is provided. 

    Args:
        likelihood: Desilike Likelihood object to run the MCMC on.
        fn_output (str, optional): Where the chains will be saved (need to have *). Defaults to 'tmp/mcmc_output_*.npy'.
        extend_chains (bool, optional): If True, it will extend the existing chains (saved in fn_output) by running new iterations. Defaults to False.
        nchains (int, optional): Number of chains to run. Defaults to 1.
        max_iterations (int, optional): Maximum number of iterations to run. Defaults to 1e5.
        check_every (int, optional): How often to check the convergence + save the current state of the chains. Defaults to 1000.

    """
    from desilike.samplers import EmceeSampler
    chains = [fn_output.replace('*', f'{i}') for i in range(nchains)] if extend_chains else nchains

    sampler = EmceeSampler(likelihood, seed=31, chains=chains, save_fn=fn_output)  
    sampler.run(max_iterations=max_iterations, check_every=check_every)

    return sampler


def get_getdist_plotter(fig_width_inch=5, fontsize=14, legend_fontsize=12, axes_labelsize=12, axes_fontsize=14, line_lables=True):
    """Wrapper around getdist to get a plotter with the desired settings."""
    from getdist import plots as gdplt

    plotter = gdplt.get_subplot_plotter()
    plotter.settings.fig_width_inch = fig_width_inch
    plotter.settings.fontsize = fontsize
    plotter.settings.legend_fontsize = legend_fontsize
    plotter.settings.axes_labelsize = axes_labelsize
    plotter.settings.axes_fontsize = axes_fontsize
    plotter.settings.line_labels = line_lables
    plotter.settings.alpha_filled_add = 0.8
    plotter.settings.legend_loc = 'upper right'
    plotter.settings.figure_legend_ncol = 1
    plotter.settings.legend_colored_text = False

    return plotter


def plot_triangle(chains, params, legend_labels=None, xlabels=[r'$f_{\rm NL}^{\rm loc}$', r'$b_1$', r'$s_{n,0}$', r'$\Sigma_s$'], 
                  contour_colors=None, filled=True, contour_ls=None, g=None, fn_output=None):
    """ Wrapper around getdist's plot_triangle to plot the contours of the different chains."""
    
    from desilike.samples import plotting
    
    if g is None: g = get_getdist_plotter()
    plotting.plot_triangle(chains, params, legend_labels=legend_labels, 
                           contour_colors=contour_colors, filled=filled, contour_ls=contour_ls, 
                           g=g, show=False, fn=None)

    if xlabels is not None:
        for i in range(len(xlabels)):
            g.subplots[len(xlabels)-1, i].set_xlabel(xlabels[i])
            if i > 0:
                g.subplots[i, 0].set_ylabel(xlabels[i])

    if fn_output is not None: plt.savefig(fn_output)
    plt.show()


def get_obervable_and_likelihood(pk, window, cov, tracer, p={'LRG': 1., 'ELG': 1., 'QSO': 1.4}, fix_fnl=False, engine='class', **kwargs):
    """_summary_

    Args:
        pk: lsstypes Mesh2SpectrumPoles observable containing the power spectrum multipoles.
        window: lsstypes WindowMatrix matches the power spectrum multipoles.
        cov: lsstypes CovarianceMatrix matches the power spectrum multipoles.
        tracer: name of the tracer used to define the parameters in the theory and avoid duplicates when combining the different observables.
        p (dict, optional): Value of p parameter. Defaults to {'LRG': 1., 'ELG': 1., 'QSO': 1.4}.
        fix_fnl (bool, optional): If true do not fit for $f_{\rm NL}^{\rm loc}$. Defaults to False.
        engine (str, optional): Solver for perturbation theory computation ('class', 'camb' or either that works in cosmoprimo). Defaults to 'class'.
        
    Kwargs:
         kwargs[f"{tracers_theo[0].split('_ell')[0]}_ell0.b1"] value used to fix one of the two b1 in the cross-correlation.

    Returns:
       observables: list of monopole and quadrupole (if used) desilike observables.
       likelihood: desilike likelihood to run profer or MCMC on.
    """
    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, PNGTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from cosmoprimo.fiducial import DESI

    tracers = tuple(tracer.split('x'))
    cross_correlation = True
    if len(tracers) == 1: 
        tracers *= 2
        cross_correlation = False

    observables = []
    zeff = []

    for ell in pk.ells:
        zeff.append(window.observable.get(ell).attrs['zeff'])   
        if cross_correlation: 
            tracers_theo = [tracer + f'_cross_ell{ell}' for tracer in tracers]
        else:
            tracers_theo = [tracer + f'_ell{ell}' for tracer in tracers]
        if not cross_correlation: tracers_theo = tracers_theo[:1]
        logger.info(f'{tracers_theo=}, {ell=}, zeff={zeff[-1]:2.4}')

        data = pk.get(ells=[ell])
        # extract only the mulitpole ell: 
        wmatrix = window.at.observable.match(data)
        covariance = cov.at.observable.at(observables='spectrum2', tracers=tracers).match(data)
        # subselect only the covariance for the cross-tracers (if tracer is a cross-tracer):
        covariance = covariance.at.observable.get(observables='spectrum2', tracers=tracers)
        
        # data, wmatrix, covariance live in this loop:
        # print(data.value().shape, wmatrix.value().shape, covariance.value().shape)
        # print(np.linalg.inv(cov.value()))

        cosmo = DESI(engine=engine)
        template = FixedPowerSpectrumTemplate(z=zeff[-1], fiducial=cosmo)
        theory = PNGTracerPowerSpectrumMultipoles(template=template, mode="b-p", tracers=tracers_theo)
        
        theory.params['fnl_loc'].update(value=0.0, fixed=fix_fnl)
        if ell == 2: theory.params[f"{'x'.join(tracers_theo)}.sn0"].update(value=0, fixed=True)
        for tracer in tracers_theo:
            theory.params[f'{tracer}.p'].update(value=p[tracer.split('_')[0]], fixed=True)

        # Don't forget to give different name for the observable in order to stack them together in the likelihood:
        observables += [TracerPowerSpectrumMultipolesObservable(name=f'pk_{tracer}', data=data, window=wmatrix, covariance=covariance, theory=theory)] 

    likelihood = ObservablesGaussianLikelihood(observables=observables, covariance=cov.at.observable.get(tracers=tracers).value(), scale_covariance=1)

    # let ell2.b1 / sigmas be derived by ell0.b1 / sigmas: 
    if len(pk.ells) > 1:
        from clustering_statistics import tools        
        for tracer in np.unique(tracers_theo):
            alpha, beta = tools.bias(1, tracer=tracer.split('_')[0], return_params=True)  # b(z) = alpha * (1 + z)**2 + beta
            factor = (alpha * (1 + zeff[1])**2 + beta) / (alpha * (1 + zeff[0])**2 + beta)
            likelihood.all_params[f"{tracer.split('_ell')[0]}_ell2.b1"].update(derived='{' + f"{tracer.split('_ell')[0]}_ell0.b1" + '}' + f' * {factor}')
            logger.warning('we neglect the redshift dependence of the damping term, for now')
            likelihood.all_params[f"{tracer.split('_ell')[0]}_ell2.sigmas"].update(derived='{' + f"{tracer.split('_ell')[0]}_ell0.sigmas" + '}')
        
    if cross_correlation: 
        if f"{tracers_theo[0].split('_ell')[0]}_ell0.b1" in kwargs:
            default_b1 = kwargs[f"{tracers_theo[0].split('_ell')[0]}_ell0.b1"] 
        else:
            default_b1 = 1
        likelihood.all_params[f"{tracers_theo[0].split('_ell')[0]}_ell0.b1"].update(value=default_b1, fixed=True)    
        likelihood.all_params[f"{tracers_theo[0].split('_ell')[0]}_ell0.sigmas"].update(value=0, fixed=True)

    likelihood()

    return observables, likelihood