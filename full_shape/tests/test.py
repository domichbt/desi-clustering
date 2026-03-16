from full_shape import tools
from full_shape.tools import generate_likelihood_options_helper, str_from_likelihood_options, get_likelihood, fill_fiducial_options, setup_logging


def test_str():
    likelihood_options = generate_likelihood_options_helper(tracer='LRG2')
    for level in [None, 1, 2, 3]:
        s = str_from_likelihood_options(likelihood_options, level=level)
        if level is None:
            assert s == 'LRG2-S2+LRG2-S3'
        elif level == 1:
            assert s == 'LRG2-S2-th-folpsD+LRG2-S3-th-folpsD+cov-holi-v1-altmtl', s
    s = str_from_likelihood_options(likelihood_options, level={'stat': 2})
    assert s == 'LRG2-S2-ell0-k0.02-0.20-0.005-ell2-k0.02-0.20-0.005+LRG2-S3-ell000-k0.02-0.12-0.005-ell202-k0.02-0.08-0.005', s

    likelihood_options = generate_likelihood_options_helper(tracer='LRG3xELG1')
    for level in [None, 1, 2, 3]:
        s = str_from_likelihood_options(likelihood_options, level=level)
        if level is None:
            assert s == 'LRG3xELG1-S2+LRG3xELG1-S3'
        elif level == 1:
            assert s == 'LRG3xELG1-S2-th-folpsD+LRG3xELG1-S3-th-folpsD+cov-holi-v1-altmtl', s
    s = str_from_likelihood_options(likelihood_options, level={'stat': 2})
    assert s == 'LRG3xELG1-S2-ell0-k0.02-0.20-0.005-ell2-k0.02-0.20-0.005+LRG3xELG1-S3-ell000-k0.02-0.12-0.005-ell202-k0.02-0.08-0.005'


def test_likelihood():
    likelihoods_options = [generate_likelihood_options_helper(tracer=tracer) for tracer in ['LRG2', 'LRG3']]
    likelihood = get_likelihood(likelihoods_options, cosmo=None, fiducial=None, cache_dir='./_cache')
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
    #test_str()
    test_likelihood()
    #test_options()