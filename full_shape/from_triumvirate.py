import argparse
import logging
from pathlib import Path
import re

import numpy as np

import lsstypes as types
from lsstypes import external, WindowMatrix

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / '_tests'
EZMOCK_INPUT_ROOT = Path('/dvs_ro/cfs/cdirs/desicollab/science/gqc/y3_fits/mockchallenge_abacus/measurements/sugiyama_basis/SecondGenMocks/EZmock/CubicBox_6Gpc')
ABACUS_INPUT_ROOT = Path('/dvs_ro/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/sugiyama_basis/SecondGenMocks/AbacusSummit/CubicBox')


def get_bk_ell_str(ell):
    return ''.join([str(_ell) for _ell in ell])


def read_triumvirate_header(fn):
    header = {}
    with open(fn, 'r') as stream:
        for line in stream:
            if not line.startswith('#'):
                break
            line = line[1:].strip()
            if ': ' in line:
                key, value = line.split(': ', 1)
                header[key] = value
    return header


def extract_seed(fn):
    match = re.search(r'seed(\d+)', Path(fn).name)
    if match is None:
        raise ValueError(f'Could not extract seed from {fn}')
    return match.group(1)


def get_dataset_output_root(output_root, dataset):
    return Path(output_root) / dataset


def convert_triumvirate_window3(tracer, zrange=(0.6, 0.8), region='NGC', add_norm=False):
    # Get theory (are keff really the theory k's??)
    imock = 0

    def get_fn(ell):
        meas_dir = Path('/dvs_ro/cfs/cdirs/desi/users/cguandal/Abacus/DR2_v1.0/LRG/z0.725/Boxes')
        return meas_dir / f'bk{get_bk_ell_str(ell)}_full_rsd_ph{imock:03d}.npy'

    ells = [(0, 0, 0), (1, 1, 0), (2, 2, 0), (0, 2, 2), (1, 1, 2), (2, 0, 2)]
    theory = convert_triumvirate_spectrum3(get_fn, ells=ells)

    theory = theory.clone(value=0. * theory.value())
    # nmodes = 1, for rebinning along the theory
    theory = theory.map(lambda pole: pole.clone(nmodes=np.ones_like(pole.values('nmodes'))))

    # Get observable
    ells = [(0, 0, 0), (2, 0, 2)]
    def get_fn(ell):
        meas_dir = Path(f'/dvs_ro/cfs/cdirs/desi/users/jaides26/window_function/cutsky_measurements/{region}/')
        if region == 'SGC':
            return meas_dir / f'bk{get_bk_ell_str(ell)}_diag_{tracer}_{region}_ph{imock:03d}.txt'
        else:  # NGC
            return meas_dir / f'bispec{get_bk_ell_str(ell)}_1024_{imock:d}_{tracer}_{region}.npy'

    observable = convert_triumvirate_spectrum3(get_fn, ells=ells)
    osize = 50
    observable = observable.select(k=slice(0, osize))

    if add_norm:
        meas_dir = Path('/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe/abacus-2ndgen-complete')
        obs = types.read(meas_dir / f'window_mesh3_spectrum_sugiyama-diagonal_poles_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_weight-default-FKP_0.h5').observable
        pole = obs.get((0, 0, 0))
        norm = pole.values('norm').mean()
        zeff, norm_zeff = pole.attrs['zeff'], pole.attrs['norm_zeff']
        observable = observable.map(lambda pole: pole.clone(norm=norm * np.ones_like(pole.values('norm')), attrs={'zeff': zeff, 'norm_zeff': norm_zeff}))

    # And window matrix value
    wmat_dir = Path('/dvs_ro/cfs/cdirs/desi/users/jaides26/window_function/wc_matrices')
    value = []
    index = np.ravel_multi_index([np.arange(osize)] * 2, dims=(osize, osize))
    for ell in ells:
        tmp = np.loadtxt(wmat_dir / f'wcmat_{get_bk_ell_str(ell)}_{tracer}_{region}_{zrange[0]:.1f}z{zrange[1]:.1f}_HF_finebin.txt')
        # Select diagonal output
        value.append(tmp[index, :])
    value = np.concatenate(value, axis=0)
    assert value.shape == (observable.size, theory.size), f'{value.shape} != {observable.size}, {theory.size}'
    return WindowMatrix(value=value, observable=observable, theory=theory)


def convert_triumvirate_spectrum2(get_fn, ells=[0, 2, 4]):
    columns = ['kbin', 'keff', 'nmodes', 'pk_raw_real', 'pk_raw_imag', 'pk_shot_real', 'pk_shot_imag']
    poles = []
    for ell in ells:
        fn = str(get_fn(ell))
        if fn.endswith('.npy'):
            state = np.load(fn, allow_pickle=True)[()]
        else:
            values = np.loadtxt(fn, unpack=True)
            state = dict(zip(columns, values))
            state['pk_raw'] = state.pop('pk_raw_real') + 1j * state.pop('pk_raw_imag')
            state['pk_shot'] = state.pop('pk_shot_real') + 1j * state.pop('pk_shot_imag')
        poles.append(state)
    return external.from_triumvirate(poles, ells=ells)


def convert_triumvirate_spectrum3(get_fn, ells=[(0, 0, 0), (2, 0, 2)]):
    columns = ['k1_bin', 'k1_eff', 'nmodes_1', 'k2_bin', 'k2_eff', 'nmodes_2', 'bk_raw_real', 'bk_raw_imag', 'bk_shot_real', 'bk_shot_imag']
    poles = []
    for ell in ells:
        fn = str(get_fn(ell))
        if fn.endswith('.npy'):
            state = np.load(fn, allow_pickle=True)[()]
        else:
            values = np.loadtxt(fn, unpack=True)
            state = dict(zip(columns, values))
            state['bk_raw'] = state.pop('bk_raw_real') + 1j * state.pop('bk_raw_imag')
            state['bk_shot'] = state.pop('bk_shot_real') + 1j * state.pop('bk_shot_imag')
        poles.append(state)
    return external.from_triumvirate(poles, ells=ells)


def convert_ezmock_box_spectrum2(
    seed,
    tracer='LRG',
    z='0.800',
    ells=(0, 2, 4),
    measurement_dir=EZMOCK_INPUT_ROOT,
    output_fn=None,
):
    """
    Convert one EZmock cubic-box power-spectrum measurement from legacy
    Triumvirate text files to an ``lsstypes.Mesh2SpectrumPoles`` object.

    Parameters
    ----------
    seed : int or str
        EZmock seed id, e.g. ``481`` or ``'0481'``.
    tracer : str, default='LRG'
        Tracer name used in the file naming convention.
    z : str, default='0.800'
        Redshift tag used in the directory and file names.
    ells : tuple, default=(0, 2, 4)
        Multipoles to read.
    measurement_dir : Path, optional
        Root directory that contains the EZmock measurements.
    output_fn : Path, optional
        If provided, write the converted spectrum to this path.
    """
    seed_str = f'{int(seed):04d}' if isinstance(seed, (int, np.integer)) else str(seed)
    pk_dir = Path(measurement_dir) / tracer / f'z{z}' / 'diag' / 'powspec'

    def get_fn(ell):
        return pk_dir / f'pk{ell}_{tracer}_z{z}_seed{seed_str}'

    header = read_triumvirate_header(get_fn(ells[0]))
    spectrum = convert_triumvirate_spectrum2(get_fn, ells=list(ells))

    attrs = {
        'source': 'triumvirate_ezmock_cubic_box',
        'tracer': tracer,
        'z': float(z),
        'seed': int(seed_str),
        'measurement_dir': str(pk_dir),
    }
    if 'Catalogue source' in header:
        attrs['catalogue_source'] = header['Catalogue source']
    if 'Box size' in header:
        attrs['boxsize'] = np.fromstring(header['Box size'].strip('[]'), sep=',')
    if 'Normalisation factor' in header:
        match = re.match(r'([0-9eE+.-]+)', header['Normalisation factor'])
        if match:
            attrs['norm'] = float(match.group(1))
    spectrum = spectrum.clone(attrs=attrs)

    if output_fn is not None:
        output_fn = Path(output_fn)
        output_fn.parent.mkdir(parents=True, exist_ok=True)
        spectrum.write(output_fn)
    return spectrum


def convert_ezmock_box_spectrum3(
    seed,
    tracer='LRG',
    z='0.800',
    ells=((0, 0, 0), (2, 0, 2)),
    measurement_dir=EZMOCK_INPUT_ROOT,
    output_fn=None,
):
    """
    Convert one EZmock cubic-box diagonal bispectrum measurement from legacy
    Triumvirate text files to an ``lsstypes.Mesh3SpectrumPoles`` object.
    """
    seed_str = f'{int(seed):04d}' if isinstance(seed, (int, np.integer)) else str(seed)
    bk_dir = Path(measurement_dir) / tracer / f'z{z}' / 'diag' / 'bispec'

    def get_fn(ell):
        return bk_dir / f'bk{get_bk_ell_str(ell)}_diag_{tracer}_z{z}_seed{seed_str}'

    header = read_triumvirate_header(get_fn(ells[0]))
    spectrum = convert_triumvirate_spectrum3(get_fn, ells=list(ells))

    attrs = {
        'source': 'triumvirate_ezmock_cubic_box',
        'tracer': tracer,
        'z': float(z),
        'seed': int(seed_str),
        'measurement_dir': str(bk_dir),
        'basis': 'sugiyama-diagonal',
    }
    if 'Catalogue source' in header:
        attrs['catalogue_source'] = header['Catalogue source']
    if 'Box size' in header:
        attrs['boxsize'] = np.fromstring(header['Box size'].strip('[]'), sep=',')
    if 'Normalisation factor' in header:
        match = re.match(r'([0-9eE+.-]+)', header['Normalisation factor'])
        if match:
            attrs['norm'] = float(match.group(1))
    spectrum = spectrum.clone(attrs=attrs)

    if output_fn is not None:
        output_fn = Path(output_fn)
        output_fn.parent.mkdir(parents=True, exist_ok=True)
        spectrum.write(output_fn)
    return spectrum


def discover_ezmock_seeds(kind, tracer='LRG', z='0.800', measurement_dir=EZMOCK_INPUT_ROOT):
    measurement_dir = Path(measurement_dir) / tracer / f'z{z}' / 'diag'
    if kind == 'mesh2_spectrum':
        pattern = f'pk0_{tracer}_z{z}_seed*'
        measurement_dir = measurement_dir / 'powspec'
    elif kind == 'mesh3_spectrum':
        pattern = f'bk000_diag_{tracer}_z{z}_seed*'
        measurement_dir = measurement_dir / 'bispec'
    else:
        raise ValueError(f'Unknown kind {kind}')
    return sorted(extract_seed(fn) for fn in measurement_dir.glob(pattern))


def get_ezmock_output_fn(kind, seed, tracer='LRG', z='0.800', output_root=DEFAULT_OUTPUT_ROOT):
    output_root = get_dataset_output_root(output_root, 'ezmock')
    seed_str = f'{int(seed):04d}' if isinstance(seed, (int, np.integer)) else str(seed)
    if kind == 'mesh2_spectrum':
        filename = f'mesh2_spectrum_poles_{tracer}_z{z}_seed{seed_str}.h5'
    elif kind == 'mesh3_spectrum':
        filename = f'mesh3_spectrum_sugiyama-diagonal_poles_{tracer}_z{z}_seed{seed_str}.h5'
    else:
        raise ValueError(f'Unknown kind {kind}')
    return output_root / tracer / f'z{z}' / filename


def run_ezmock_batch(todo, tracer='LRG', z='0.800', seeds=None, all_seeds=False, output_root=DEFAULT_OUTPUT_ROOT):
    if all_seeds == (seeds is not None):
        raise ValueError('Provide exactly one of seeds or all_seeds')

    successes = 0
    failures = []

    for kind in todo:
        kind_seeds = sorted({f'{int(seed):04d}' for seed in seeds}) if seeds is not None else discover_ezmock_seeds(kind=kind, tracer=tracer, z=z)
        logger.info('Converting %s for %d seed(s)', kind, len(kind_seeds))
        for seed in kind_seeds:
            output_fn = get_ezmock_output_fn(kind=kind, seed=seed, tracer=tracer, z=z, output_root=output_root)
            try:
                if kind == 'mesh2_spectrum':
                    convert_ezmock_box_spectrum2(seed=seed, tracer=tracer, z=z, output_fn=output_fn)
                elif kind == 'mesh3_spectrum':
                    convert_ezmock_box_spectrum3(seed=seed, tracer=tracer, z=z, output_fn=output_fn)
                else:
                    raise ValueError(f'Unknown kind {kind}')
                successes += 1
                logger.info('Wrote %s', output_fn)
            except Exception as exc:
                failures.append((kind, seed, exc))
                logger.exception('Failed to convert %s seed%s', kind, seed)

    logger.info('Finished EZmock conversion: %d success(es), %d failure(s)', successes, len(failures))
    return successes, failures


def get_abacus_output_fn(kind, imock, tracer='LRG', z='0.500', output_root=DEFAULT_OUTPUT_ROOT):
    output_root = get_dataset_output_root(output_root, 'abacus')
    imock_str = f'{int(imock):03d}' if isinstance(imock, (int, np.integer)) else str(imock)
    if kind == 'mesh2_spectrum':
        filename = f'mesh2_spectrum_poles_{tracer}_z{z}_ph{imock_str}.h5'
    elif kind == 'mesh3_spectrum':
        filename = f'mesh3_spectrum_sugiyama-diagonal_poles_{tracer}_z{z}_ph{imock_str}.h5'
    elif kind == 'mesh3_spectrum_full':
        filename = f'mesh3_spectrum_sugiyama_poles_{tracer}_z{z}_ph{imock_str}.h5'
    else:
        raise ValueError(f'Unknown kind {kind}')
    return output_root / tracer / f'z{z}' / filename


def run_abacus_batch(todo, tracer='LRG', z='0.500', imocks=None, output_root=DEFAULT_OUTPUT_ROOT):
    imocks = ['000'] if imocks is None else [f'{int(imock):03d}' for imock in imocks]
    successes = 0
    failures = []

    for kind in todo:
        logger.info('Converting %s for %d Abacus mock(s)', kind, len(imocks))
        for imock in imocks:
            output_fn = get_abacus_output_fn(kind=kind, imock=imock, tracer=tracer, z=z, output_root=output_root)
            try:
                if kind == 'mesh2_spectrum':
                    meas_dir = ABACUS_INPUT_ROOT / tracer / f'z{z}' / 'diag' / 'powspec'
                    spectrum = convert_triumvirate_spectrum2(lambda ell: meas_dir / f'pk{ell:d}_{tracer}_z{z}_ph{imock}')
                elif kind == 'mesh3_spectrum':
                    meas_dir = ABACUS_INPUT_ROOT / tracer / f'z{z}' / 'diag' / 'bispec'
                    spectrum = convert_triumvirate_spectrum3(lambda ell: meas_dir / f'bk{get_bk_ell_str(ell)}_diag_{tracer}_z{z}_ph{imock}')
                elif kind == 'mesh3_spectrum_full':
                    meas_dir = ABACUS_INPUT_ROOT / tracer / f'z{z}' / 'full' / 'bispec_2d' / 'lin'
                    spectrum = convert_triumvirate_spectrum3(lambda ell: meas_dir / f'bk{get_bk_ell_str(ell)}_full_{tracer}_z{z}_ph{imock}')
                else:
                    raise ValueError(f'Unknown kind {kind}')
                output_fn.parent.mkdir(parents=True, exist_ok=True)
                spectrum.write(output_fn)
                successes += 1
                logger.info('Wrote %s', output_fn)
            except Exception as exc:
                failures.append((kind, imock, exc))
                logger.exception('Failed to convert %s ph%s', kind, imock)

    logger.info('Finished Abacus conversion: %d success(es), %d failure(s)', successes, len(failures))
    return successes, failures


def get_window_output_fn(tracer='LRG', zrange=(0.6, 0.8), region='NGC', output_root=DEFAULT_OUTPUT_ROOT):
    output_root = get_dataset_output_root(output_root, 'window')
    filename = f'window_mesh3_spectrum_sugiyama-diagonal_poles_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}.h5'
    return output_root / tracer / filename


def run_window_batch(todo, tracer='LRG', zrange=(0.6, 0.8), regions=None, add_norm=False, output_root=DEFAULT_OUTPUT_ROOT):
    regions = ['NGC', 'SGC'] if regions is None else list(regions)
    successes = 0
    failures = []

    for kind in todo:
        if kind != 'window_mesh3_spectrum':
            raise ValueError(f'Unsupported window kind {kind}')
        logger.info('Converting %s for %d region(s)', kind, len(regions))
        for region in regions:
            output_fn = get_window_output_fn(tracer=tracer, zrange=zrange, region=region, output_root=output_root)
            try:
                window = convert_triumvirate_window3(tracer=tracer, zrange=zrange, region=region, add_norm=add_norm)
                output_fn.parent.mkdir(parents=True, exist_ok=True)
                window.write(output_fn)
                successes += 1
                logger.info('Wrote %s', output_fn)
            except Exception as exc:
                failures.append((kind, region, exc))
                logger.exception('Failed to convert %s %s', kind, region)

    logger.info('Finished window conversion: %d success(es), %d failure(s)', successes, len(failures))
    return successes, failures


def parse_args():
    parser = argparse.ArgumentParser(description='Convert legacy Triumvirate products to lsstypes.')
    parser.add_argument('--dataset', choices=['ezmock', 'abacus', 'window'], required=True)
    parser.add_argument('--todo', nargs='+', choices=['mesh2_spectrum', 'mesh3_spectrum', 'mesh3_spectrum_full', 'window_mesh3_spectrum'], required=True, help='One or more statistics to convert.')
    parser.add_argument('--tracer', default='LRG')
    parser.add_argument('--z', default='0.800')
    parser.add_argument('--seed', nargs='+', help='One or more EZmock seeds, e.g. 0001 0481.')
    parser.add_argument('--all-seeds', action='store_true', help='Convert all discovered seeds for the requested statistics.')
    parser.add_argument('--imock', nargs='+', help='One or more Abacus mock indices, e.g. 000 001.')
    parser.add_argument('--region', nargs='+', choices=['NGC', 'SGC'], help='Window regions to convert.')
    parser.add_argument('--zrange', nargs=2, type=float, metavar=('ZMIN', 'ZMAX'), help='Redshift range for window conversion.')
    parser.add_argument('--add-norm', action='store_true', help='Add norm metadata when converting window products.')
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    valid_todo = {
        'ezmock': {'mesh2_spectrum', 'mesh3_spectrum'},
        'abacus': {'mesh2_spectrum', 'mesh3_spectrum', 'mesh3_spectrum_full'},
        'window': {'window_mesh3_spectrum'},
    }
    if not set(args.todo).issubset(valid_todo[args.dataset]):
        parser.error(f'Unsupported --todo for dataset {args.dataset}: {args.todo}')

    if args.dataset == 'ezmock' and args.all_seeds == (args.seed is not None):
        parser.error('Provide exactly one of --seed or --all-seeds for EZmock.')
    if args.dataset != 'ezmock' and (args.seed is not None or args.all_seeds):
        parser.error('--seed/--all-seeds are only valid for --dataset ezmock.')
    if args.dataset != 'abacus' and args.imock is not None:
        parser.error('--imock is only valid for --dataset abacus.')
    if args.dataset != 'window' and (args.region is not None or args.zrange is not None or args.add_norm):
        parser.error('--region/--zrange/--add-norm are only valid for --dataset window.')
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(levelname)s:%(message)s')

    if args.dataset == 'ezmock':
        _, failures = run_ezmock_batch(
            todo=args.todo,
            tracer=args.tracer,
            z=args.z,
            seeds=args.seed,
            all_seeds=args.all_seeds,
            output_root=args.output_root,
        )
    elif args.dataset == 'abacus':
        _, failures = run_abacus_batch(
            todo=args.todo,
            tracer=args.tracer,
            z=args.z,
            imocks=args.imock,
            output_root=args.output_root,
        )
    elif args.dataset == 'window':
        _, failures = run_window_batch(
            todo=args.todo,
            tracer=args.tracer,
            zrange=tuple(args.zrange) if args.zrange is not None else (0.6, 0.8),
            regions=args.region,
            add_norm=args.add_norm,
            output_root=args.output_root,
        )
    else:
        raise ValueError(f'Unsupported dataset {args.dataset}')
    if failures:
        raise SystemExit(1)
