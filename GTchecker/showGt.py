#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tifffile
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",     type=str,    required=True,         help="the original dataset path")
parser.add_argument("--name_file_path",  type=str,    required=True,         help="the image name file path")
parser.add_argument("--output_dir",      type=str,    default="./gtShow",    help="the dir to save shown gt images")
parser.add_argument("--cmap_method" ,    type=str,    default="jet",         help="the color method for cmap")
parser.add_argument("--data_name",       type=str,    default="pos",         help="the data type, vaihingen or potsdam")
args = parser.parse_args()


def save_img(fpath, outname):
    img_arr = tifffile.imread(fpath)
    if args.data_name == 'pos':
        # corner crop with size 5000x5000
        img_arr = img_arr[0:5000, 0:5000]
    plt.imsave(os.path.join(args.output_dir, outname), img_arr, cmap=args.cmap_method)


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.name_file_path, 'r') as f:
        flist = [i[:-1] for i in f.readlines()]

    for fname in flist:
        fpath = os.path.join(args.dataset_dir, fname)
        outname = "shown_" + fname
        try:
            save_img(fpath, outname)
        except IOError:
            print("cannot open %s"%(fpath))


if __name__ == '__main__':
    main()
