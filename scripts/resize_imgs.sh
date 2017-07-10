#!/usr/bin/env bash

# This scripts find images in a directory tree, resizes them and puts them into a new directory,
# while preserving the directory structure.
#
# Usage:
#
#   ./resize_imgs.sh input_dir output_dir
#

input_root_dir=$1
output_root_dir=$2

# output dir as an absolute path
if [ ! -d $output_root_dir ]; then
    mkdir -p $output_root_dir
fi

cd $output_root_dir
output_root_dir=$(pwd)
cd -

current_dir=$(pwd)


cd $input_root_dir
for img_path in $(find . -iname *.png); do
    echo "processing $img_path"

    output_path=$output_root_dir/$img_path
    output_dir=$(dirname $output_path)

    echo "processing $output_path"
    if [ ! -d $output_dir ]; then
        mkdir -p $output_dir
    fi

    convert -resize 621x187 $img_path $output_path
done

cd -