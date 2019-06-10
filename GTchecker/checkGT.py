#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tifffile
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str,  required=True,        help="the original dataset path")
parser.add_argument("--data_name",     type=str,  required=True,        help="Vaihingen or Potsdam: ['vai', 'pos']")
parser.add_argument("--output_dir",    type=str,  default="./gtShow",   help="the dir to save shown gt images")
parser.add_argument("--cmap_method" ,  type=str,  default="jet",        help="the color method for cmap")
parser.add_argument("--check_mode",    type=str,  default="vis",        help="visual or statistics: ['vis', 'sta']")
args = parser.parse_args()


def read_img(fpath):
    img_arr = tifffile.imread(fpath)
    if args.data_name == 'pos':
        # corner crop with size 5000x5000
        img_arr = img_arr[0:5000, 0:5000]
    return img_arr


def save_img(img_arr, outname):
    save_path = os.path.join(args.output_dir, outname)
    plt.imsave(save_path, img_arr, cmap=args.cmap_method)


def get_flist(name_file_path):
    with open(name_file_path, 'r') as f:
        flist = [name[:-1] for name in f.readlines()]
    return flist


def save_gt(flist):
    for fname in tqdm(flist):
        fpath = os.path.join(args.dataset_dir, fname)
        outname = "shown_" + fname
        try:
            img_arr = read_img(fpath)
            save_img(img_arr, outname)
        except IOError:
            print("cannot open {}".format(fpath))


def study_gt(flist):
    # create temp data container
    glob_arr = None

    if args.data_name == 'vai':
        glob_arr = np.array([])

        for fname in tqdm(flist):
            fpath = os.path.join(args.dataset_dir, fname)
            img_arr = read_img(fpath).astype(np.float32)
            glob_arr = np.append(glob_arr, img_arr.flatten())
    elif args.data_name == 'pos':
        flen = len(flist)
        glob_arr = np.zeros([flen, 5000, 5000]).astype(np.float32)

        for it, fname in enumerate(tqdm(flist)):
            fpath = os.path.join(args.dataset_dir, fname)
            glob_arr[it] = np.nan_to_num(read_img(fpath))
        glob_arr = glob_arr.flatten()
    else:
        pass

    print("Getting data info ...")
    glob_min, glob_max, glob_mean = map(lambda arr: np.round(arr, decimals=2), [np.min(glob_arr), np.max(glob_arr), np.mean(glob_arr)])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    n, bins, patches = plt.hist(x=glob_arr, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.85)
    plt.xlabel('Height range')
    plt.ylabel('Frequency')
    plt.title("Histogram of {} $\\rightarrow$ $\mu:$ {:.2f}, min: {:.2f}, max: {:.2f}".format(args.data_name, glob_mean, glob_min, glob_max))
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(os.path.join(args.output_dir, '{}_info.png'.format(args.data_name)))


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    name_file_path = ""
    if args.data_name == 'pos':
        name_file_path = "./pos_dsm_all.txt"
    elif args.data_name == 'vai':
        name_file_path = "./vai_dsm_all.txt"
    else:
        raise ValueError("Panic: Unvalid Data Name")

    flist = get_flist(name_file_path)

    if args.check_mode == 'vis':
        print("Saving colorized ground truth ...")
        save_gt(flist)
    elif args.check_mode == 'sta':
        print("Checking ...")
        study_gt(flist)
    else:
        raise ValueError("Panic: Unvalid check mode")


if __name__ == '__main__':
    main()
