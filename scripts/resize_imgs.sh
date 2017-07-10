#!/usr/bin/env bash

# This scripts finds images in a directory tree, resizes them and put them into a new directory
# by preserving the directory structure.
#
# Usage:
# ./resize_imgs.sh input_dir output_dir
#

input_dir=$1
output_dir=$2

echo $input_dir
echo $output_dir

for img_path in $(find $input_dir -iname *.png); do
    echo "processing $img_path"
    output_path=$output_dir/$img_path

    if [ ! -d $output_path ]; then
        mkdir -p $output_path
    fi

    convert -resize 621x187 $input_dir/$img_path
done