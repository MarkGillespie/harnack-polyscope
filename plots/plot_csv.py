#!/usr/bin/env python3

# Generate convergence plot

import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, LogLocator
from draw_loglog_slope import draw_loglog_slope
from label_lines import label_lines

import os
import sys
import signal
import random

from math import pi

def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}
matplotlib.rc('font', **font)
matplotlib.rc('pdf', fonttype=42)

# Style options
# sns.set(font="Linux Biolinum O", font_scale=8./12., rc={"xtick.bottom" : True, "ytick.left" : True, "pdf.use14corefonts": True})
sns.set(font="Linux Biolinum O",
        font_scale=8./12.,
        rc={"xtick.bottom" : True, "ytick.left" : True})

keenan_purple = "#1b1f8a"
alt_color = "#1f8a1b"
sad_color = "#ff0000"

def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="file to process")
    args = parser.parse_args()


    df = pd.read_csv(args.data, header=0, sep=',',engine='python')
    print("Done loading data")
    print(df.columns)
    print(df.columns[0])

    line_plot = True
    if line_plot:
        scale = 2
        fig, ax = plt.subplots(figsize=(3.377 * scale,1.75 * scale))
        ax.plot(df[df.columns[0]], df[df.columns[1]], color=keenan_purple)

        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        fig.set_size_inches(3.38, 3)

        plt.tight_layout()
        # fig.savefig(os.path.join(out_dir,'_plot.pdf'))

        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
