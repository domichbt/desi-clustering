#!/bin/bash
# script to copy forFA{imock}.fits from desica scratch to its respective mock directory
# bash copy_forFA.sh GLAM-Uchuu_v2 150 199
for ((i=$2;i<=$3;i++ ))
do
        echo copying $SCRATCH/DA2/mocks/$1/forFA$i.fits 
        cp $SCRATCH/DA2/mocks/$1/forFA$i.fits /global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/$1/
done