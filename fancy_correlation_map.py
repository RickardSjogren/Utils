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
import scipy.spatial.distance as sp_dist
import scipy.cluster.hierarchy as sp_hierarchy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, patches, gridspec

__author__ = 'Rickard Sjögren'
__copyright__ = 'Copyright (C) 2016 Rickard Sjögren'
__licence__ = 'MIT'
__version__ = '1.2'


def correlation_matrix(*data, on_columns=True):
    """ Calculate correlation matrix.
    
    If one data matrix is provided in `data` the auto-correlation
    matrix will be calculated of rows/columns. If two data matrices
    are provided the correlations between rows/columns of `data[0]`
    and rows/columns of `data[1]` will be calculated.
    
    Parameters
    ----------
    *data : tuple[array_like]
        Tuple of one or two data matrices.
    on_columns : bool
        If True, calculate correlations between columns, else rows.
    
    Returns
    -------
    correlations : array_like
        Correlation matrix.
    significance : array_like
        Matrix with significance levels of `correlations`.
    """
    try:
        first_data, second_data = data
    except ValueError:
        first_data = second_data = data[0]
    
    if on_columns:
        first_data = first_data.T
        second_data = second_data.T
        
    n_1, m_1 = first_data.shape
    n_2, m_2 = second_data.shape
    
    if m_1 != m_2:
        raise ValueError('dimensions does not match')
    
    correlations = np.zeros((n_1, n_2))
    significance = np.zeros((n_1, n_2))
    
    for i, row in enumerate(first_data):
        for j, other_row in enumerate(second_data):
            missing = np.logical_or(np.isnan(row), np.isnan(other_row))            
            corr, p = stats.pearsonr(row[~missing], other_row[~missing])
            correlations[i, j] = corr
            significance[i, j] = p
            
    return correlations, significance
    


def make_fancy_heatmap(correlations, significance, labels, alpha=.05,
                       draw_numbers=False, mark_significant=False,
                       mark_insignificant=False, no_frame=None, title=None,
                       ax=None, xlabel_loc='top', ylabel_loc='left'):
    """ Draw a fancy heatmap and return figure and axis-instance.
    
    Parameters
    ----------
    correlations : array_like
        n x m-matrix of correlation-values
    significance : array_like
        n x m-matrix of correlation p-values
    labels : Sequence, tuple[Sequence]
        Sequence of labels to use. If n != m, tuple of sequences with labels.
    alpha : float
        Significance value to use.
    draw_numbers : bool
        If True, correlation numbers will be drawn in upper quadrant if n = m.
    mark_significant : bool
        If True, significant correlations will be marked with square.
    mark_insignifican : bool
        If True, insignificant correlations will be marked with cross.
    no_frame : bool, None
        If True remove frame. Default is True when n = m, else False.
    title : str, None
        If provided, figure title will be set to `title`.
    xlabel_loc : {'top', 'bottom'}
        Where to draw X-axis labels, default 'top'
    ylabel_loc : {'left', 'right'}
        Where to draw Y-axis labels, default 'left'
    Returns
    -------
    matplotlib.pyplot.Figure
    matplotlib.pyplot.Axes
    """
    n, m = correlations.shape

    if n < m:
        correlations = correlations.T
        significance = significance.T
        n, m = m, n
        labels = labels[::-1]

    if ax is None:
        f, ax = plt.subplots(dpi=600)
        f.set_size_inches(.35 * m, .35 * n)
    else:
        f = None

    is_square = n == m

    if not is_square:                
        y_labels, x_labels = labels
        no_frame = False if no_frame is None else no_frame
    else:
        y_labels = x_labels = labels
        no_frame = (True and not draw_numbers) if no_frame is None else no_frame
        print('No frame is: {}'.format(no_frame))
    
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
        inner_range = range(row, n) if is_square else range(m)        

        for col in inner_range:
            corr = correlations[row, col]
            radius = np.sqrt(abs(corr)) / 2  # Area scales with correlation.
            color = corr_colors[row, col]
            
            if draw_numbers and is_square:
                x = row
                y = col
                
                # Add text-label with numeric value when off-diagonal.
                if row != col:            
                    label = '{:.2f}'.format(corr)
                    ax.annotate(label, (col + .5, row + .5),
                                horizontalalignment='center',
                                verticalalignment='center', size='x-small')
            else:
                x = col
                y = row
            
            circle = plt.Circle((x + .5, y + .5), .95 * radius, color=color)
            ax.add_artist(circle)
            
            # Draw cross if not statistically significant.
            if significance[row, col] > alpha and mark_insignificant:
                ax.plot([x + .2, x+.8], [y + .2, y+.8], 'k-', linewidth=1)
                ax.plot([x + .2, x+.8], [y+.8, y + .2], 'k-', linewidth=1)
                
            # Draw square if statistically significant.
            if significance[row, col] < alpha and mark_significant:
                if row == col and is_square:
                    continue
                rec = patches.Rectangle((x, y), 1, 1, facecolor='none',
                                        edgecolor='k', alpha=1)
                ax.add_patch(rec)
    
    # Adjust ticks and axes.            
    ax.set_xticks(np.arange(m + 1) + .5)
    ax.set_yticks(np.arange(n + 1) + .5)
    ax.set_xlim([0, m])
    ax.set_ylim([0, n])
    
    if (is_square and xlabel_loc is None) or xlabel_loc =='top':
        ax.xaxis.tick_top()
    elif xlabel_loc =='bottom':
        ax.xaxis.tick_bottom()
    
    if (not (draw_numbers and is_square)) or ylabel_loc == 'right':
        ax.yaxis.tick_right()
    elif ylabel_loc == 'left':
        ax.yaxis.tick_left()
    
    ax.set_xticklabels(x_labels, rotation=90 if is_square else 270)
    ax.set_yticklabels(y_labels)
    
    if title is not None:
        ax.set_title(title)
    
    # Remove frame
    if no_frame:
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
    

def make_fancy_correlationmap(correlations, significance, labels,
                              *args, **kwargs):
    """

    Parameters
    ----------
    correlations
    args
    kwargs

    Returns
    -------

    """
    m, n = correlations.shape

    if n < m:
        correlations = correlations.T
        significance = significance.T
        n, m = m, n
        labels = labels[::-1]

    row_dist = sp_dist.squareform(sp_dist.pdist(correlations))
    column_dist = sp_dist.squareform(sp_dist.pdist(correlations.T))

    column_linkage = sp_hierarchy.linkage(column_dist, method='ward')
    row_linkage = sp_hierarchy.linkage(row_dist, method='ward')

    f = plt.figure()
    gs = gridspec.GridSpec(6, 6)
    axes = np.array([
        [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1:])],
        [plt.subplot(gs[1:, 0]), plt.subplot(gs[1:, 1:])]
    ])

    f.set_size_inches(.35 * m, .35 * n)

    row_dendrogram = sp_hierarchy.dendrogram(column_linkage, orientation='right',
                                             ax=axes[1, 0], color_threshold=0,
                                            link_color_func=lambda x: 'black')
    col_dendrogram = sp_hierarchy.dendrogram(row_linkage, orientation='top',
                                             ax=axes[0, 1], color_threshold=0,
                                             link_color_func=lambda x: 'black')

    col_ind = row_dendrogram['leaves']
    row_ind = col_dendrogram['leaves']
    
    data = correlations.copy()[:, col_ind][row_ind, :]
    sigs = significance.copy()[:, col_ind][row_ind, :]

    if m == n:
        labels = np.array(labels)[row_ind]
        axes[1, 0].invert_yaxis()
    else:
        labels = [
            np.array(labels[1])[row_ind],
            np.array(labels[0])[col_ind],
        ]
    x_loc = 'bottom' if (m == n and kwargs['draw_numbers']) else None
    y_loc = 'right' if (m == n and kwargs['draw_numbers']) else None
    
    make_fancy_heatmap(data, sigs, labels, *args, ax=axes[1, 1],
                       xlabel_loc=x_loc, ylabel_loc=y_loc, **kwargs)

    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[1, 0].axis('off')

    f.subplots_adjust(hspace=0, wspace=0)
    return f, axes

    
if __name__ == '__main__':
    import argparse
    import pandas as pd
    
    def required_length(nmin, nmax):
        class RequiredLength(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if not nmin<=len(values)<=nmax:
                    msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                        f=self.dest,nmin=nmin,nmax=nmax)
                    raise argparse.ArgumentTypeError(msg)
                setattr(args, self.dest, values)
        return RequiredLength
    
    parser = argparse.ArgumentParser(
        description='Draw fancy correlation heatmap'
    )
    parser.add_argument('csvfile', type=str, nargs='+',
                        action=required_length(1, 2),
                        help='csv-file(s) containing data to plot')
    parser.add_argument('--alpha', type=float, default=.05,
                        help='significance level')
    parser.add_argument('--mark_significant', const=True, action='store_const',
                        default=False, help=('If this flag is set, significant'
                        ' correlations will be marked with square.'))
    parser.add_argument('--mark_insignificant', const=True, action='store_const',
                        default=False, help=('If this flag is set, insignificant'
                        ' correlations will be marked with cross.'))
    parser.add_argument('--use_rows', const=True, action='store_const',
                        default=False, 
                        help=('if this flag is set, correlations'
                              ' will be calculated on rows'))
    parser.add_argument('-o', type=str,
                        help=('output file, if not set will'
                              ' use name of csv-file instead'))
    parser.add_argument('--draw_numbers', const=True, action='store_const',
                        default=False,
                        help=('If this flag is set, correlations as numbers'
                              ' will be drawn in upper triangle.'))
    parser.add_argument('--figure_title', default=None, 
                        help='optional plot title')
    parser.add_argument('--cluster', default=False, action='store_true',
                        help='If this flag is set, ')
    
    args = parser.parse_args()
    
    if not all(f.endswith('csv') for f in args.csvfile):
        print('Input must be in csv-format')
        sys.exit(1)
        
    if not 0 < args.alpha < 1:
        print('alpha must be between 0 and 1')
        sys.exit(1)
        
    data = [pd.DataFrame.from_csv(f) for f in args.csvfile]
    arrays = [df.values for df in data]
    
    corr, p = correlation_matrix(*arrays, on_columns=not args.use_rows)
    
    if len(data) == 1:
        labels = data[0].columns if not args.use_rows else data[0].index
        columns = labels
        index = labels
    else:
        labels = [d.columns if not args.use_rows else d.index for d in data]
        index, columns = labels
        m, n = corr.shape
        if n < m:
            index, columns = columns, index
        
        
    if args.cluster:
        if not (args.draw_numbers or len(data) == 2):
            msg = ('Clustermap requires two datasets or that numbers'
                    ' are drawn. It looks horrible otherwise.')
            sys.exit(msg)
        plotter = make_fancy_correlationmap
    else:
        plotter = make_fancy_heatmap
    
    f, ax = plotter(
        corr, p, labels, args.alpha, args.draw_numbers,
        mark_significant=args.mark_significant,
        mark_insignificant=args.mark_insignificant,
        title=args.figure_title
    )
    
    if len(data) == 1 and not args.o:
        filename = args.csvfile[0][:-4] + '_correlation'
    elif len(data) == 2 and not args.o:
        filename = '_'.join(f[:-4] for f in args.csvfile)  + '_correlation'
    else:
        filename = args.o
        
    f.savefig(filename + '.png', dpi=600, bbox_inches='tight')
    pd.DataFrame(corr, index=index, columns=columns).to_csv(filename + '.csv')
    pd.DataFrame(p, index=index, columns=columns).to_csv(filename + '_pvalues.csv')
