from matplotlib import pyplot as plt

import lsstypes as types
from clustering_statistics import tools


def get_means_covs(kind, versions, tracer, zrange, region, stats_dir, project='', ells=(0,2,4), rebin=1):
    means, covs = {}, {}
    for version in versions:
        kw = {'tracer': tracer}
        for name in ['version', 'weight', 'cut', 'auw', 'extra']:
            kw[name] = versions[version][name]
        if 'ELG' in kw['tracer']:
            if 'complete' in kw['version']:
                kw['tracer'] = 'ELG_LOP'
            elif 'data' in kw['version']:
                kw['tracer'] = 'ELG_LOPnotqso'
        if 'mesh3' in kind:
            kw['basis'] = 'sugiyama-diagonal'
            kw['auw'] = False
        fns = tools.get_stats_fn(kind=kind, stats_dir=stats_dir, project=project, zrange=zrange, region=region, **kw, imock='*')
        stats = list(map(types.read, fns))
        # print(fns[0])
        if 'particle2_correlation' in kind:
            stats = [stat.project(ells=ells) for stat in stats]
            means[version] = types.mean(stats).select(s=slice(0, None, rebin))
        else:
            means[version] = types.mean(stats).select(k=slice(0, None, rebin))
        if len(stats) > 1:
            covs[version] = types.cov(stats).at.observable.match(means[version])
        else:
            covs[version] = None
    return means, covs


def plot_stats(kind, versions, tracer, zrange, region, stats_dir, project='', ells=(0,2,4), rebin=1, reference=None, ylim=(-1.5, 1.5),
               figure=None, ax_col=0, linestyles=None, lw=2, colors=None, scaling='kpk', save_fn=None, title=None):
    if reference is None:
        # use first item from versions as reference
        reference = next(iter(versions))
    if linestyles is None: linestyles = dict(zip(versions, ['-']*len(versions)))
    if colors is None: colors = dict(zip(versions, [f'C{i:d}' for i in range(len(versions))]))
    if figure is None:
        fig, lax = plt.subplots(len(ells) * 2, figsize=(6, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1] * len(ells)})
    else:
        fig, axes = figure
        if axes.ndim == 1:
            lax = axes
        else:
            lax = axes[:, ax_col]
    k_exp = 1 if scaling == 'kpk' else 0
    s_exp = 2
    if 'mesh2_spectrum' in kind:
        means, covs = get_means_covs(kind, versions, tracer, zrange, region, stats_dir, project=project, rebin=rebin)
        versions = list(means)
        if title is None:
            lax[0].set_title(f'{tracer} in {region} {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        else:
            lax[0].set_title(title)
        for ill, ell in enumerate(ells):
            ax = lax[2 * ill]
            if scaling == 'kpk':
                ax.set_ylabel(rf'$k P_{ell:d}(k)$ [$(\mathrm{{Mpc}}/h)^2$]')
            if scaling == 'loglog':
                ax.set_ylabel(rf'$P_{ell:d}(k)$ [$(\mathrm{{Mpc}}/h)^3$]')
                ax.set_yscale('log')
                ax.set_xscale('log')
            for iversion, version in enumerate(versions):
                if ell not in means[version].ells: continue
                pole = means[version].get(ell)
                value = pole.coords('k')**k_exp * pole.value().real
                if 'data' in version or version == reference:
                    std = pole.coords('k')**k_exp * covs[reference].at.observable.get(ell).std().real
                    ax.fill_between(pole.coords('k'), value - std, value + std, color=colors[version], alpha=0.2)
                ax.plot(pole.coords('k'), value, color=colors[version], linestyle=linestyles[version], label=version, lw=lw)
            if ill == 0: ax.legend(frameon=False, ncol=1)
            ax.grid(True)
            ax = lax[2 * ill + 1]
            ax.set_ylabel(rf'$\Delta P_{ell:d} / \sigma(k)$')
            ax.grid(True)
            ax.set_ylim(*ylim)
            for iversion, version in enumerate(versions):
                if 'data' in version or version == reference: continue
                pole = means[version].get(ell)
                # std = covs[reference].at.observable.get(ell).std().real
                std = covs[reference].at.observable.get(ell).std().real
                # print(tracer,zrange,version,kind,ell,std.min())
                ax.plot(pole.coords('k'), (pole.value() - means[reference].get(ell).value()).real / std, color=colors[version], linestyle=linestyles[version], lw=lw)
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    elif 'mesh3_spectrum_sugiyama-diagonal' in kind:
        means, covs = get_means_covs('mesh3_spectrum', versions, tracer, zrange, region, stats_dir, project=project, rebin=rebin)
        versions = list(means)
        if title is None:
            lax[0].set_title(f'{tracer} in {region} {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        else:
            lax[0].set_title(title)
        for ill, ell in enumerate(ells):
            ax = lax[2 * ill]
            if scaling == 'kpk':
                ax.set_ylabel(rf'$k^2 B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}}(k, k)$ [$(\mathrm{{Mpc}}/h)^6$]')
            if scaling == 'loglog':
                ax.set_ylabel(rf'$B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}}(k, k)$ [$(\mathrm{{Mpc}}/h)^4$]')
                ax.set_yscale('log')
                ax.set_xscale('log')

            for iversion, version in enumerate(versions):
                if ell not in means[version].ells: continue
                pole = means[version].get(ell)
                x = pole.coords('k')[..., 0]
                value = (x**2)**k_exp * pole.value().real
                if 'data' in version or version == reference:
                    std = (x**2)**k_exp * covs[reference].at.observable.get(ell).std().real
                    ax.fill_between(x, value - std, value + std, color=colors[version], alpha=0.2)
                ax.plot(x, value, color=colors[version], linestyle=linestyles[version], label=version, lw=lw)
                if ill == 0: ax.legend(frameon=False, ncol=2)
            ax.grid(True)
            ax = lax[2 * ill + 1]
            ax.set_ylabel(rf'$\Delta B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}} / \sigma(k)$')
            ax.grid(True)
            ax.set_ylim(*ylim)
            for iversion, version in enumerate(versions):
                if 'data' in version or version == reference: continue
                pole = means[version].get(ell)
                std = covs[reference].at.observable.get(ell).std().real
                # print(tracer,zrange,version,kind,ell,std.min())
                x = pole.coords('k')[..., 0]
                ax.plot(x, (pole.value() - means[reference].get(ell).value()).real / std, color=colors[version], linestyle=linestyles[version], lw=lw)
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    elif 'particle2_correlation' in kind:
        means, covs = get_means_covs(kind, versions, tracer, zrange, region, stats_dir, project=project, ells=ells, rebin=rebin)
        versions = list(means)
        if title is None:
            lax[0].set_title(f'{tracer} in {region} {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        else:
            lax[0].set_title(title)
        for ill, ell in enumerate(ells):
            ax = lax[2 * ill]
            ax.set_ylabel(rf'$s^2 \xi_{ell:d}(s)$ [$(\mathrm{{Mpc}}/h)^2$]')
            for iversion, version in enumerate(versions):
                if ell not in means[version].ells: continue
                pole = means[version].get(ell)
                value = pole.coords('s')**s_exp * pole.value().real
                if 'data' in version or version == reference:
                    std = pole.coords('s')**s_exp * covs[reference].at.observable.get(ell).std().real
                    ax.fill_between(pole.coords('s'), value - std, value + std, color=colors[version], alpha=0.2)
                ax.plot(pole.coords('s'), value, color=colors[version], linestyle=linestyles[version], label=version, lw=lw)
            if ill == 0: ax.legend(frameon=False, ncol=1)
            ax.grid(True)
            ax = lax[2 * ill + 1]
            ax.set_ylabel(rf'$\Delta P_{ell:d} / \sigma(k)$')
            ax.grid(True)
            ax.set_ylim(*ylim)
            for iversion, version in enumerate(versions):
                if 'data' in version or version == reference: continue
                pole = means[version].get(ell)
                # std = covs[reference].at.observable.get(ell).std().real
                std = covs[reference].at.observable.get(ell).std().real
                # print(tracer,zrange,version,kind,ell,std.min())
                ax.plot(pole.coords('s'), (pole.value() - means[reference].get(ell).value()).real / std, color=colors[version], linestyle=linestyles[version], lw=lw)
        lax[-1].set_xlabel(r'$s$ [$h^{-1}\mathrm{Mpc}$]')

    if save_fn and figure is None:
        plt.tight_layout()
        fig.savefig(save_fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.show()