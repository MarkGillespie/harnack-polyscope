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

def converged(x):
    return x <= 0.01 or x >= 4. * pi - 0.01

def trimmed_relative_errs(vals, epsilon):
    target = 4. * pi if (abs(4. * pi - vals.values[-1]) < 1.) else 0.
    relative_errors = abs(target - vals ) / abs(target - vals[0] )
    convergence_start = relative_errors[relative_errors < epsilon].index[0]
    return relative_errors.iloc[convergence_start:]

# Compute run statistics
def compute_stats(errs):
    # https://www.pythoncharts.com/python/line-chart-with-confidence-interval/
    stats =  errs.agg(['mean', 'std', 'count'], axis=1)

    # Calculate a confidence interval as well.
    # https://www.alchemer.com/resources/blog/how-to-calculate-confidence-intervals/
    # h_stats['ci'] = 1.96 * h_stats['std'] / np.sqrt(h_stats['count']) # 95%
    stats['ci'] = 2.576 * stats['std'] / np.sqrt(stats['count']) # 99%
    stats['ci_lower'] = stats['mean'] - stats['ci']
    stats['ci_upper'] = stats['mean'] + stats['ci']

    return stats
    # print(h_stats)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('harnack_data', help="file to process")
    parser.add_argument('sphere_data', help="file to process")
    # parser.add_argument('--record_files', action='store_true')
    # parser.add_argument('--max_files', type=int, default = -1, help="read in at most max_files many of the input files")
    # parser.add_argument('--merged_files', default = "", help="csv of dataset")
    args = parser.parse_args()


    # configuration
    # matplotlib.rcParams.update({'font.size': 8})

    # Place results here
    file_dir, file_name = os.path.split(args.harnack_data)
    out_dir = os.path.join(file_dir, f"{file_name[:-4]}_analysis")
    ensure_dir_exists(out_dir)

    h_df = pd.read_csv(args.harnack_data, header=0, sep=',',engine='python')
    # h_df['iteration'] = np.arange(len(h_df)) # record iteration number with each step
    h_df = h_df.drop(['grad_termination', 'overstepping', 'newton_steps'], axis=1) # drop parameter info
    times = [c for c in h_df if c.endswith('_ts')]
    h_times_df = h_df[times].ffill() # grab time data in separate df
    h_df = h_df.drop(times, axis=1) # drop all time info (columns ending in _ts)
    h_df = h_df.ffill() # fill missing values with last present value in column

    s_df = pd.read_csv(args.sphere_data, header=0, sep=',',engine='python')
    # s_df['iteration'] = np.arange(len(s_df)) # record iteration number with each step
    s_df = s_df.drop(['grad_termination', 'overstepping', 'newton_steps'], axis=1) # drop parameter info
    print(s_df)
    times = [c for c in s_df if c.endswith('_ts')]
    s_times_df = s_df[times].ffill() # grab time data in separate df
    s_df = s_df.drop(times, axis=1) # drop all time info (columns ending in _ts)
    s_df = s_df.ffill() # fill missing values with last present value in column

    print("Done loading data")

    plot_function_err = True
    plot_distance_err = True

    if plot_function_err:
        scale = 2
        fig, ax = plt.subplots(figsize=(3.377 * scale,1.75 * scale))

        h_converged_cols = [col for i, col in enumerate(h_df.columns) if converged(h_df.iloc[-1, i])]
        h_df_converged_errs = h_df[h_converged_cols].map(lambda x : min(abs(x), abs(4. * pi - x)))
        h_df_converged_rel_errs = h_df_converged_errs.apply(lambda col : col / col.iloc[0])
        h_stats = compute_stats(h_df_converged_rel_errs)

        s_converged_cols = [col for i, col in enumerate(s_df.columns) if abs(s_df.iloc[-1, i]) < 0.01]
        s_df_converged_errs = s_df[s_converged_cols].map(lambda x : abs(x))
        s_df_converged_rel_errs = s_df_converged_errs.apply(lambda col : col / col.iloc[0])
        s_stats = compute_stats(s_df_converged_rel_errs);

        ax.plot(1+s_stats.index, s_stats['mean'], color=alt_color, alpha=0.75, label="Sphere tracing")
        ax.fill_between(
            1+s_stats.index, s_stats['ci_lower'], s_stats['ci_upper'], color=alt_color, alpha=.15)

        ax.plot(1+h_stats.index, h_stats['mean'], color=keenan_purple, alpha=0.75, label="Harnack tracing")
        ax.fill_between(
            1+h_stats.index, h_stats['ci_lower'], h_stats['ci_upper'], color=keenan_purple, alpha=.15)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("iterations")
        ax.set_ylabel("relative error in function value")
        ax.set_title(f"Convergence Rate (function value)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([1e-4, 1.5])
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.tick_params(color='0.7', axis='x', which='minor')
        ax.tick_params(color='0.7', axis='y', which='minor')

        label_lines(plt.gca().get_lines())

        draw_loglog_slope(fig, ax, origin=(5, 0.0001), width_inches=1, slope=1, inverted=False, color='k')

        plt.subplots_adjust(bottom=0.2)

        fig.set_size_inches(3.38, 3)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir,'harnack_function_convergence_plot.pdf'))

        plt.show()
        plt.close()

    if plot_distance_err:
        scale = 2
        fig, ax = plt.subplots(figsize=(3.377 * scale,1.75 * scale))

        h_converged_cols = [i for i, col in enumerate(h_df.columns) if converged(h_df.iloc[-1, i])]
        # in each column, subtract times from final times to get time err (i.e. distance err)
        h_times_df_converged_errs = h_times_df.iloc[:,h_converged_cols].apply(lambda col: col.iloc[-1] - col)
        h_times_df_converged_rel_errs = h_times_df_converged_errs.apply(lambda col : col / col.iloc[0])
        h_times_stats = compute_stats(h_times_df_converged_rel_errs)

        s_converged_cols = [i for i, col in enumerate(s_df.columns) if converged(s_df.iloc[-1, i])]
        # in each column, subtract times from final times to get time err (i.e. distance err)
        s_times_df_converged_errs = s_times_df.iloc[:,s_converged_cols].apply(lambda col: col.iloc[-1] - col)
        print(s_times_df_converged_errs)

        s_df_converged_errs = s_df.iloc[:,s_converged_cols].apply(lambda col: col.iloc[-1] - col)
        print(s_df_converged_errs)

        s_times_df_converged_rel_errs = s_times_df_converged_errs.apply(lambda col : col / col.iloc[0])
        s_times_stats = compute_stats(s_times_df_converged_rel_errs)

        ax.plot(1+s_times_stats.index, s_times_stats['mean'], color=alt_color, alpha=0.75, label="Sphere tracing")
        ax.fill_between(
            1+s_times_stats.index, s_times_stats['ci_lower'], s_times_stats['ci_upper'], color=alt_color, alpha=.15)

        ax.plot(1+h_times_stats.index, h_times_stats['mean'], color=keenan_purple, alpha=0.75, label="Harnack tracing")
        ax.fill_between(
            1+h_times_stats.index, h_times_stats['ci_lower'], h_times_stats['ci_upper'], color=keenan_purple, alpha=.15)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("iterations")
        ax.set_ylabel("relative error in distance")
        ax.set_title(f"Convergence Rate (distance)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([1e-4, 1.5])
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs = np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.tick_params(color='0.7', axis='x', which='minor')
        ax.tick_params(color='0.7', axis='y', which='minor')

        label_lines(plt.gca().get_lines())

        draw_loglog_slope(fig, ax, origin=(5, 0.0001), width_inches=1, slope=1, inverted=False, color='k')

        plt.subplots_adjust(bottom=0.2)

        fig.set_size_inches(3.38, 3)

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir,'harnack_distance_convergence_plot.pdf'))

        plt.show()
        plt.close()

if __name__ == "__main__":
    main()
