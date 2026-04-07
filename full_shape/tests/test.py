from pathlib import Path

import numpy as np
import pytest

from full_shape import tools
from full_shape.tools import generate_likelihood_options_helper, str_from_likelihood_options, str_from_options, get_likelihood, fill_fiducial_options, setup_logging
from full_shape.job_scripts.validation_abacus_mocks import _build_likelihoods_options, _build_run_options, _get_parser


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

    options['cosmology'] = {'model': 'base', 'template': 'direct'}
    options = fill_fiducial_options(options)
    s = str_from_options(options, level=None)
    assert s == 'cosmo-base_LRG3xELG1-S2+LRG3xELG1-S3', s


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


def test_validation_abacus_mocks_theory_model_options():
    for theory_model in ['folpsD', 'folpsEFT', 'reptvelocileptors']:
        stats = ['mesh2_spectrum']
        likelihoods = _build_likelihoods_options(
            stats=stats,
            tracers=['LRG1'],
            version='abacus-2ndgen-dr2-complete',
            covariance='holi-v3-altmtl',
            stats_dir=Path('/tmp'),
            theory_model=theory_model,
        )
        assert len(likelihoods) == 1
        observables = likelihoods[0]['observables']
        assert [observable['stat']['kind'] for observable in observables] == stats
        assert all(observable['theory']['model'] == theory_model for observable in observables)


def test_validation_abacus_mocks_nchains_option():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        nchains=4,
    )
    assert options['sampler']['nchains'] == 4


def test_validation_abacus_mocks_cosmo_model_option():
    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        cosmo_model='base',
    )
    assert options['cosmology']['model'] == 'base'

    options = _build_run_options(
        stats=['mesh2_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/tmp'),
        theory_model='folpsD',
        cosmo_model='base_ns-fixed',
    )
    assert options['cosmology']['model'] == 'base_ns-fixed'


def test_validation_abacus_mocks_parser_accepts_nchains():
    parser = _get_parser()
    args = parser.parse_args(['--todo', 'sample', '--nchains', '4'])
    assert args.todo == ['sample']
    assert args.nchains == 4


def test_validation_abacus_mocks_parser_accepts_cosmo_params():
    parser = _get_parser()
    args = parser.parse_args(['--cosmo_params', 'base'])
    assert args.cosmo_params == 'base'

    args = parser.parse_args(['--cosmo_params', 'base_ns-fixed'])
    assert args.cosmo_params == 'base_ns-fixed'


def test_validation_abacus_mocks_parser_defaults_cosmo_params_to_base():
    parser = _get_parser()
    args = parser.parse_args([])
    assert args.cosmo_params == 'base'


def test_validation_abacus_mocks_parser_help_describes_cosmo_params():
    parser = _get_parser()
    help_text = parser.format_help()
    assert '--cosmo_params' in help_text
    assert 'base varies h, omega_cdm, omega_b, logA, n_s' in help_text
    assert 'base_ns-fixed varies h, omega_cdm, omega_b, logA' in help_text


def test_folpsEFT_nuisance_priors_define_refs():
    mesh2_params = tools._get_default_theory_nuisance_priors(
        model='folpsEFT',
        stat='mesh2_spectrum',
        prior_basis='physical_aap',
        b3_coev=True,
        sigma8_fid=0.8,
    )
    for name in ['b1p', 'b2p', 'bsp', 'alpha0p', 'alpha2p', 'alpha4p', 'sn0p', 'sn2p']:
        assert 'ref' in mesh2_params[name], name
        assert mesh2_params[name]['ref']['dist'] == 'norm'
    assert mesh2_params['b3p']['fixed'] is True
    assert 'ref' not in mesh2_params['b3p']
    assert mesh2_params['X_FoG_pp']['fixed'] is True
    assert 'ref' not in mesh2_params['X_FoG_pp']

    mesh3_params = tools._get_default_theory_nuisance_priors(
        model='folpsEFT',
        stat='mesh3_spectrum',
        prior_basis='physical_aap',
        sigma8_fid=0.8,
    )
    for name in ['b1p', 'b2p', 'bsp', 'c1p', 'c2p', 'Pshotp', 'Bshotp']:
        assert 'ref' in mesh3_params[name], name
        assert mesh3_params[name]['ref']['dist'] == 'norm'
    assert mesh3_params['X_FoG_bp']['fixed'] is True
    assert 'ref' not in mesh3_params['X_FoG_bp']


def test_folpsEFT_sampler_start_is_full_rank():
    from desilike.samplers import EmceeSampler

    options = {}
    options['likelihoods'] = _build_likelihoods_options(
        stats=['mesh2_spectrum', 'mesh3_spectrum'],
        tracers=['LRG1'],
        version='abacus-2ndgen-dr2-complete',
        covariance='holi-v3-altmtl',
        stats_dir=Path('/global/cfs/cdirs/desicollab/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base'),
        theory_model='folpsEFT',
    )
    options['cosmology'] = {'template': 'direct'}
    options = fill_fiducial_options(options)
    likelihood = get_likelihood(options['likelihoods'], cosmology_options=options['cosmology'], cache_dir='./_cache')
    likelihood()

    sampler = EmceeSampler(likelihood, seed=42, nwalkers=32)
    start = sampler._get_start()[0]
    assert np.isfinite(start).all()
    assert np.linalg.matrix_rank(start - start.mean(axis=0, keepdims=True)) == start.shape[-1]


def test_validation_abacus_mocks_reptvelocileptors_rejects_mesh3():
    with pytest.raises(ValueError, match='reptvelocileptors.*mesh2_spectrum'):
        _build_likelihoods_options(
            stats=['mesh2_spectrum', 'mesh3_spectrum'],
            tracers=['LRG1'],
            version='abacus-2ndgen-dr2-complete',
            covariance='holi-v3-altmtl',
            stats_dir=Path('/tmp'),
            theory_model='reptvelocileptors',
        )


if __name__ == '__main__':

    setup_logging()
    test_covariance()
    test_str()
    test_likelihood_bao()
    test_likelihood_full_shape()
    test_covariance()
    test_options()
