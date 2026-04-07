#!/bin/bash
# script to copy rename measurements folders. Known dubious mocks in particular.
# bash rename_dubious_folders.sh

base_dir=/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/base/holi-v3-altmtl/
mapfile -t mocks_list < dubious_holi-v3-altmtl.txt
for imock in "${mock_list[@]}"; do
    echo "Renaming $imock"
    mv $base_dir/mock$imock/ $base_dir/dubious_mock_$imock
done