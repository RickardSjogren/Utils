#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains two functions to calculate and draw a correlation
map to show pairwise Pearson correlations and significances of correlation.

See figure 2 in:
PLoS Negl Trop Dis. 2016 Mar 4;10(3):e0004480.
doi: 10.1371/journal.pntd.0004480. eCollection 2016.

Can be run as command-line script.
"""
from __future__ import print_function
import sys
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

__author__ = 'Rickard Sjögren'
__copyright__ = 'Copyright (C) 2016 Rickard Sjögren'
__licence__ = 'MIT'
__version__ = '1.0'


def correlation_matrix(data, on_columns=True):
    """ Calculate correlation-matrix along significances of 
    correlations of data.
    
    Parameters
    ----------
    data : array_like
        Data matrix
    on_columns : bool
        If True, calculate correlations between columns, else rows.
    
    Returns
    -------
    correlations : array_like
        Square correlation matrix.
    significance : array_like
        Square matrix with significance levels.
    """
    data = data.T if on_columns else data
    n = len(data)
    
    correlations = np.zeros((n, n))    
    significance = np.zeros((n, n))
    for i, row in enumerate(data):
        for j, other_row in enumerate(data):
            corr, p = stats.pearsonr(row, other_row)
            correlations[i, j] = corr
            significance[i, j] = p
            
    return correlations, significance


def make_fancy_heatmap(correlations, significance, labels, alpha=.05):
    """ Draw a fancy heatmap and return figure and axis-instance.
    
    Parameters
    ----------
    correlations : array_like
        n x n-matrix of correlation-values
    significance : array_like
        n x n-matrix of correlation p-values
    alpha : float
        Significance value to use.
    
    Returns
    -------
    matplotlib.pyplot.Figure
    matplotlib.pyplot.Axes
    """
    f, ax = plt.subplots(dpi=600)
    n = len(correlations)    
    f.set_size_inches(.35 * n, .35 * n)    
    
    norm = colors.Normalize(-1, 1)
    
    # Custom colormap with nicer red and blue.
    cmap = colors.LinearSegmentedColormap.from_list('custom', [
        (0, (0.09, 0.36, 0.62)),
        (.5, 'white'),
        (1, (0.404, 0, 0.12))    
    ])
    color_map = plt.cm.ScalarMappable(norm, cmap)
    corr_colors = color_map.to_rgba(correlations)
    
    for row in range(n):
        for col in range(row, n):
            corr = correlations[row, col]
            radius = np.sqrt(abs(corr)) / 2  # Area scales with correlation.
            color = corr_colors[row, col]
            circle = plt.Circle((row + .5, col + .5), .95 * radius, color=color)
            ax.add_artist(circle)
            
            # Add text-label with numeric value when off-diagonal.
            if row != col:            
                label = '{:.2f}'.format(corr)
                ax.annotate(label, (col + .5, row + .5),
                            horizontalalignment='center',
                            verticalalignment='center', size='x-small')
            
            # Draw cross if not stastically significant.
            if significance[row, col] > alpha:
                ax.plot([row + .2, row+.8], [col + .2, col+.8], 'k-', linewidth=1)
                ax.plot([row + .2, row+.8], [col+.8, col + .2], 'k-', linewidth=1)
    
    # Adjust ticks and axes.            
    ax.set_xticks(np.arange(n + 1) + .5)
    ax.set_yticks(np.arange(n + 1) + .5)
    ax.set_xlim([0, n])
    ax.set_ylim([0, n])
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    
    # Remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set minor ticks for use with grid.
    minor_locator = ticker.AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    
    # Remove all tick-marks.
    all_ticks = ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks()
    all_ticks += ax.xaxis.get_minor_ticks() + ax.yaxis.get_minor_ticks()
    for tic in all_ticks:
        tic.tick1On = tic.tick2On = False
    
    ax.grid(which='minor', linestyle='-', color='gray')
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')                        
    
    return f, ax
    
    
if __name__ == '__main__':
    import argparse
    import pandas
    
    parser = argparse.ArgumentParser(
        description='Draw fancy correlation heatmap'
    )
    parser.add_argument('csvfile', type=str,
                        help='csv-file containing data to plot')
    parser.add_argument('--alpha', type=float, default=.05,
                        help='significance level')
    parser.add_argument('--use_rows', const=True, action='store_const',
                        default=False, 
                        help=('if this flag is set, correlations'
                              ' will be calculated on rows'))
    parser.add_argument('-o', type=str,
                        help=('output file, if not set will'
                              ' use name of csv-file instead'))
    args = parser.parse_args()
    
    if not args.csvfile.endswith('csv'):
        print('Input must be in csv-format')
        sys.exit(1)
        
    if not 0 < args.alpha < 1:
        print('alpha must be between 0 and 1')
        sys.exit(1)

    data = pandas.DataFrame.from_csv(args.csvfile)
    corr, p = correlation_matrix(data.values, on_columns=not args.use_rows)
    labels = data.columns if not args.use_rows else data.index
    f, ax = make_fancy_heatmap(corr, p, labels, args.alpha)
    
    filename = args.csvfile[:-4] + '_correlation.png' if not args.o else args.o
    f.savefig(filename, dpi=600, bbox_inches='tight')