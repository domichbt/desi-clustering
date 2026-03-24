from pathlib import Path

import numpy as np

from full_shape import tools
from full_shape.tools import generate_likelihood_options_helper, str_from_likelihood_options, str_from_options, get_likelihood, fill_fiducial_options, setup_logging


def test_str():
    likelihood_options = generate_likelihood_options_helper(tracer='LRG2')
    for level in [None, 1, 2, 3]:
        s = str_from_likelihood_options(likelihood_options, level=level)
        if level is None:
            assert s == 'LRG2-S2+LRG2-S3'
        elif level == 1:
            assert s == 'LRG2-S2-th-folpsD+LRG2-S3-th-folpsD+cov-mock-holi-v1-altmtl', s
    s = str_from_likelihood_options(likelihood_options, level={'stat': 2})
    assert s == 'LRG2-S2-ell0-k0.02-0.20-0.005-ell2-k0.02-0.20-0.005+LRG2-S3-ell000-k0.02-0.12-0.005-ell202-k0.02-0.08-0.005', s

    likelihood_options = generate_likelihood_options_helper(tracer='LRG3xELG1')
    for level in [None, 1, 2, 3]:
        s = str_from_likelihood_options(likelihood_options, level=level)
        if level is None:
            assert s == 'LRG3xELG1-S2+LRG3xELG1-S3'
        elif level == 1:
            assert s == 'LRG3xELG1-S2-th-folpsD+LRG3xELG1-S3-th-folpsD+cov-mock-holi-v1-altmtl', s
    s = str_from_likelihood_options(likelihood_options, level={'stat': 2})
    assert s == 'LRG3xELG1-S2-ell0-k0.02-0.20-0.005-ell2-k0.02-0.20-0.005+LRG3xELG1-S3-ell000-k0.02-0.12-0.005-ell202-k0.02-0.08-0.005'

    options = {}
    options['likelihoods'] = [likelihood_options]
    options = fill_fiducial_options(options)
    s = str_from_options(options, level=None)
    assert s == 'cosmo-base_ns-fixed_LRG3xELG1-S2+LRG3xELG1-S3', s


def test_likelihood_full_shape():
    options = {}
    options['likelihoods'] = [generate_likelihood_options_helper(tracer=tracer) for tracer in ['LRG2', 'LRG3']]
    for template in ['direct', 'shapefit']:
        options['cosmology'] = {'template': template}
        options = fill_fiducial_options(options)
        likelihood = get_likelihood(options['likelihoods'], cosmology_options=options['cosmology'], cache_dir='./_cache')
        likelihood()
        print(likelihood.varied_params)
        if template == 'direct':
            assert 'h' in likelihood.varied_params
        elif template == 'shapefit':
            assert 'df' in likelihood.varied_params


def test_likelihood_bao():
    options = {}
    stats_dir = Path('/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')
    for template in ['bao', 'direct'][:1]:
        options['likelihoods'] = [generate_likelihood_options_helper(stats=['recon_particle2_correlation'], tracer=tracer, version='data-dr2-v1.1', stats_dir=stats_dir, emulator=template == 'direct') for tracer in ['LRG2']]
        for likelihood_options in options['likelihoods']:
            likelihood_options['covariance'] = {'source': 'rascalc', 'version': 'data-dr2-v1.1', 'stats_dir': stats_dir}
        options['cosmology'] = {'template': template}
        options = fill_fiducial_options(options)
        likelihood = get_likelihood(options['likelihoods'], cosmology_options=options['cosmology'], cache_dir='./_cache', cache_mode='w')
        assert np.isfinite(likelihood())
        if template == 'direct':
            assert 'h' in likelihood.varied_params
        elif template == 'bao':
            assert 'qpar' in likelihood.varied_params
            from desilike.profilers import MinuitProfiler
            profiler = MinuitProfiler(likelihood, seed=42)
            profiler.maximize()
            likelihood(**profiler.profiles.bestfit.choice(input=True, index='argmax'))
            print(profiler.profiles.to_stats(tablefmt='pretty'))
            likelihood.likelihoods[0].observables[0].plot(fn='./_tests/plot.png')
            likelihood.likelihoods[0].observables[0].plot_bao(fn='./_tests/plot_bao.png')


def test_covariance():
    options = {}
    options['likelihoods'] = [generate_likelihood_options_helper(stats=['mesh2_spectrum'], tracer=tracer) for tracer in ['LRG3']]
    for likelihood_options in options['likelihoods']:
        likelihood_options['covariance'] = {'source': 'jaxpower', 'version': 'abacus-2ndgen-complete'}
    options = fill_fiducial_options(options)
    likelihood = get_likelihood(options['likelihoods'], cache_dir='./_cache')
    likelihood()


def test_options():
    options = {}
    options['likelihoods'] = [generate_likelihood_options_helper(tracer=tracer) for tracer in ['LRG2', 'LRG3']]
    options = fill_fiducial_options(options)
    options2 = tools._base_type_options(options)
    fn = '_tests/config.yaml'
    tools.write_options(fn, options)
    options3 = tools.read_options(fn)
    assert options3 == options2


if __name__ == '__main__':

    setup_logging()
    test_covariance()
    exit()
    test_str()
    test_likelihood_bao()
    test_likelihood_full_shape()
    test_covariance()
    test_options()