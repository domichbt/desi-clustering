#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# copy_mocks_by_id.sh
# This script copies measurements performed on mocks from a source directory 
# into subdirectories based on the mock identifier contained in their filenames.
# Filenames are assumed to follow the pattern:
# 
#   basefilename_imock.extension
# 
# For each matching file:
# 
# 1. A directory named 'mock{imock}' is created (if it does not exist).
# 2. The file is copied into that directory.
# 3. The copied file has the identifier 'imock' removed from its filename.
# Note: A pattern can be passed as a third argument such that it will only copy 
# files containing this pattern. This is useful for copying a particular statistic.
# 
# examples: 
# 1. Copy all measurements in source_directory:
#    ./copy_by_id.sh /source_directory /destination_directory ""
# 2. Copy only power spectrum measurements:
#    ./copy_by_id.sh /source_directory /destination_directory "mesh2_spectrum"
# -----------------------------------------------------------------------------

src_dir="$1"
dst_dir="$2"
pattern="$3"

for file in "$src_dir"/*"$pattern"*; do
    [[ -f "$file" ]] || continue
    filename="${file##*/}"

    [[ "$filename" == .* ]] && continue

    name="${filename%.*}"
    ext=""
    [[ "$filename" == *.* ]] && ext=".${filename##*.}"

    id="${name##*_}"
    base="${name%_*}"

    mkdir -p "$dst_dir/mock$id"
    cp "$file" "$dst_dir/mock$id/$base$ext"
done
