"""
Run with:
```bash
salloc -N 1 -C cpu -t 04:00:00 --qos interactive
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh new
srun -n 18 python make_complete_mocks.py
```
"""

import os
from pathlib import Path

import numpy as np
from mpi4py import MPI

from clustering_statistics import tools
from clustering_statistics.tools import setup_logging


def make_complete_catalogs(tracer='LRG', imock=128, version='glam-uchuu-v1-altmtl', output_cat_dir=Path(os.getenv('SCRATCH')) / 'measurements', nran=18):
    mpicomm = MPI.COMM_WORLD
    remove_columns = ['INDWEIGHT', 'POSITION']  # not needed
    catalog_options = dict(version=version, tracer=tracer, zrange=None, weight='default', imock=imock)
    # Pass complete = {} to create a complete data catalog on-the-fly
    # And reshuffle to store the new (complete) data catalog to resample redshifts from in cache
    seed = 100 * imock
    complete, reshuffle = {'seed': seed}, {}
    try:
        data = tools.read_clustering_catalog(kind='data', complete=complete, reshuffle=reshuffle, keep_columns=True, region='ALL', **catalog_options, mpicomm=MPI.COMM_SELF)
    except IOError:
        return
    del data[remove_columns]
    if mpicomm.rank == 0:
        for region in ['NGC', 'SGC']:
            data_fn = tools.get_catalog_fn(kind='data', cat_dir=output_cat_dir, region=region, **catalog_options)
            data[tools.select_region(data['RA'], data['DEC'], region)].write(data_fn, group='LSS')

    for iran in range(nran):
        irank = iran % mpicomm.size
        if mpicomm.rank == irank:
            # Can be combined with expand
            expand = {'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=[iran])[0]}
            randoms = tools.read_clustering_catalog(kind='randoms', complete=complete, reshuffle=reshuffle | {'seed': seed + iran}, expand=expand, keep_columns=True, concatenate=False, nran=[iran], region='ALL', **catalog_options, mpicomm=MPI.COMM_SELF)[0]
            del randoms[remove_columns]
            for region in ['NGC', 'SGC']:
                randoms_fn = tools.get_catalog_fn(kind='randoms', cat_dir=output_cat_dir, nran=[iran], region=region, **catalog_options)[0]
                randoms[tools.select_region(randoms['RA'], randoms['DEC'], region)].write(randoms_fn, group='LSS')


if __name__ == '__main__':

    setup_logging()

    tracers = ['LRG', 'ELG_LOPnotqso', 'QSO']
    imocks = 150 + np.arange(50)
    nran = 18
    version = 'glam-uchuu-v2-altmtl'
    if 'altmtl' in version:
        complete_version = str(version).replace('altmtl','') 
    else:
        complete_version = version
    # output_cat_dir = Path(os.getenv('SCRATCH')) / 'clustering_catalogs' / f'{version}_complete'
    output_cat_dir = tools.base_stats_dir / 'auxiliary_data' / f'{complete_version}complete'

    for tracer in tracers:
        for imock in imocks:
            full_output_cat_dir = output_cat_dir / f'complete{imock:d}/loa-v1/mock{imock:d}/LSScats/'
            make_complete_catalogs(tracer=tracer, imock=imock, version=version, output_cat_dir=full_output_cat_dir, nran=nran)