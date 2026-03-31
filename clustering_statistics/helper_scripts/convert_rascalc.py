from pathlib import Path

import numpy as np
from lsstypes import CovarianceMatrix, WindowMatrix, GaussianLikelihood, Mesh2SpectrumPole, Mesh2SpectrumPoles, Count2CorrelationPole, Count2CorrelationPoles, ObservableLeaf, ObservableTree

from clustering_statistics import tools


def _convert_covariance(covariance, with_attrs=False):

    from cosmoprimo.fiducial import DESI
    from cosmoprimo import constants
    fiducial = DESI()

    def _get_attrs(observable):
        if with_attrs:
            return {name: np.float64(observable.attrs[name]) for name in ['zeff']}
        return {}

    def _get_fiducial(parameter, zeff=None):
        toret = {}
        if zeff is not None:
            DM_over_rd_fid = fiducial.comoving_angular_distance(zeff) / fiducial.rs_drag
            DH_over_rd_fid = (constants.c / 1e3) / (100. * fiducial.efunc(zeff)) / fiducial.rs_drag
            DV_over_rd_fid = DM_over_rd_fid**(2. / 3.) * DH_over_rd_fid**(1. / 3.) * zeff**(1. / 3.)
            FAP_fid = DM_over_rd_fid / DH_over_rd_fid
            if parameter == 'qpar':
                toret['DH_over_rd_fid'] = DH_over_rd_fid
            if parameter == 'qper':
                toret['DM_over_rd_fid'] = DM_over_rd_fid
            if parameter == 'qiso':
                toret['DV_over_rd_fid'] = DV_over_rd_fid
        return toret

    def convert_spectrum(observable):
        spectrum = []
        attrs = _get_attrs(observable)
        for ell, k, edges, nmodes, value in zip(observable.projs, observable._x, observable._edges, observable._weights, observable._value):
            edges = np.column_stack([edges[:-1], edges[1:]])
            spectrum.append(Mesh2SpectrumPole(k=k, k_edges=edges, nmodes=nmodes, num_raw=value, ell=ell))
        return Mesh2SpectrumPoles(spectrum, attrs=attrs)

    def convert_correlation(observable):
        correlation = []
        attrs = _get_attrs(observable)
        for ell, s, edges, weights, value in zip(observable.projs, observable._x, observable._edges, observable._weights, observable._value):
            edges = np.column_stack([edges[:-1], edges[1:]])
            correlation.append(Count2CorrelationPole(s=s, s_edges=edges, value=value, ell=ell))
        return Count2CorrelationPoles(correlation, attrs=attrs)

    def convert_compressed(observable):
        leaves, names = [], []
        attrs = _get_attrs(observable)
        for proj, value in zip(observable.projs, observable._value):
            names.append(proj)
            assert proj in ['qiso', 'qap', 'qpar', 'qper', 'df', 'dm']
            leaf = ObservableLeaf(value=np.atleast_1d(value), attrs=_get_fiducial(proj, **attrs))
            leaves.append(leaf)
        return ObservableTree(leaves, parameters=names, attrs=attrs)

    observables, names = [], []
    for observable in covariance.observables():
        if observable.name == 'power':
            names.append('spectrum')
            observables.append(convert_spectrum(observable))
        elif observable.name == 'correlation':
            names.append('correlation')
            observables.append(convert_correlation(observable))
        elif observable.name == 'correlation-recon':
            names.append(observable.name.replace('-', ''))
            observables.append(convert_correlation(observable))
        elif observable.name in ['shapefit', 'bao-recon']:
            names.append(observable.name.replace('-', ''))
            observables.append(convert_compressed(observable))
        elif observable.name in ['shapefit+bao-recon']:
            names.append('shapefit')
            observables.append(convert_compressed(observable))
        else:
            raise NotImplementedError(observable.name)
    if len(observables) > 1:
        observable = ObservableTree(observables, observables=names)
    else:
        observable = observables[0]
    value = covariance.view()
    covariance = CovarianceMatrix(value=value, observable=observable)
    return covariance


def convert_covariance(dp_fn, dr_fn, tracer):
    from desilike.observables import ObservableCovariance
    covariance = ObservableCovariance.load(dp_fn)
    covariance = _convert_covariance(covariance)
    covariance = covariance.clone(observable=ObservableTree([covariance.observable], observables=['correlation2recon'], tracers=[(tools.get_simple_tracer(tracer),) * 2]))
    covariance.write(dr_fn)


if __name__ == '__main__':

    dp_dir = Path('/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/cov_2pt/rascalc/v1.1/')

    list_zrange = [('BGS_BRIGHT-21.35', (0.1, 0.4)),
                   ('LRG', (0.4, 0.6)),
                   ('LRG', (0.6, 0.8)),
                   ('LRG', (0.8, 1.1)),
                   ('LRG+ELG_LOPnotqso', (0.8, 1.1)),
                   ('ELG_LOPnotqso', (1.1, 1.6)),
                   ('QSO', (0.8, 2.1))]
    for tracer, zrange in list_zrange:
        for region in ['GCcomb', 'NGC', 'SGC']:
            dp_covariance_fn = f'covariance_correlation_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin.npy'
            stats_dir = '/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe'
            kw = dict(stats_dir=stats_dir, weight='default-FKP', version='data-dr2-v1.1', tracer=tracer, region=region, zrange=zrange)
            covariance_fn = tools.get_stats_fn(kind='covariance_particle2_correlation_rascalc', **kw)
            convert_covariance(dp_dir / dp_covariance_fn, covariance_fn, tracer=tracer)
            
            dp_covariance_fn = f'covariance_correlation-recon_{tracer}_{region}_z{zrange[0]:.1f}-{zrange[1]:.1f}_default_FKP_lin.npy'
            covariance_fn = tools.get_stats_fn(kind='covariance_recon_particle2_correlation_rascalc', **kw)
            convert_covariance(dp_dir / dp_covariance_fn, covariance_fn, tracer=tracer)