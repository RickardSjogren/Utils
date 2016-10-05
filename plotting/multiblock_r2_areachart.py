#!/usr/bin/python
# -*- coding: utf-8 -*-
""" This module contains a function to plot an area chart with
one-dimensional silhouette at top.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Standard journal image size and dpi.
DPI = 300
HEIGHT = 2625 / DPI
WIDTH = 2250 / DPI


def area_chart_silhouette(area_data, silhouette, names, colors=None,
                          xlabel=None, indexes=None, tick_labels=None, **kwargs):
    """ Make stacked area chart with silhouette at top.

    Parameters
    ----------
    area_data : array_like
        (n x k) column array of data to plot in area chart.
    silhouette : array_like
        length n, data to plot as silhouette.
    names : list[str]
        Legend labels.
    xlabel : str, optional
        Axis x-labels.
    colors : list
        length k, sequence of colors to use.
    xlabel : str, optional
        X-axis label.
    indexes : list[int], optional
        Sequence of tick-location indexes.
    tick_labels : list[str], optional
        Sequence of tick-labels-
    **kwargs
        Key-word arguments passed to :py:`matplotlib.pyplot.figure`

    Returns
    -------
    matplotlib.pyplot.Figure
        Figure instance.
    tuple[matplotlib.pyplot.Axes]
        Length 2. Area chart axis and silhouette axis.
    matplotlib.pyplot.Legend
        Plot legend.
    """
    gs = gridspec.GridSpec(4, 4, hspace=0)

    if not kwargs.get('figsize', False):
        kwargs['figsize'] = (WIDTH, HEIGHT*.5)

    f = plt.figure(**kwargs)
    ax = f.add_subplot(gs[1:, :])
    ax2 = f.add_subplot(gs[0, :])

    n = area_data.shape[1]
    colors = colors if colors is not None else sns.color_palette('Blues', n)

    ax.stackplot(np.arange(len(area_data)), *area_data.T, labels=names,
                 colors=colors)
    ax2.stackplot(np.arange(len(area_data)), silhouette, colors=['black'])

    legend = ax.legend(bbox_to_anchor=(1.5, 1))

    ax2.set_xlim((0, len(area_data)))
    ax.set_xlim((0, len(area_data)))

    ax.set_ylabel('R2')
    ax2.axis('off')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if indexes is not None:
        ax.set_xticks(indexes)
    if tick_labels is not None:
        ax.set_xticklabels(tick_labels)

    f.tight_layout()
    f.subplots_adjust(right=.7)

    return f, (ax, ax2), legend


if __name__ == '__main__':
    import argparse
    import scipy.io
    import sys

    help_text = """Area chart with silhouette.

    Mandatory variables:
    * AreaData : Columns contain data to plot in area chart.
    * Silhouette : Single column with silhouette data.
    * AreaNames : Names for each area in area chart.

    Optional variables:
    * TickIndexes : Zero-indexed x-axis tick locations.
    * TickLabels : X-axis tick labels.
    * BarColors : Colors of each area."""
    parser = argparse.ArgumentParser(help_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mat',
                        help="""mat-file with data.""")
    parser.add_argument('--xlabel', help='X-axis label to use', default=None)
    parser.add_argument('-o', '--output', help='Output filename',
                        default='areachart.tiff')
    parser.add_argument('--width', help='image width in pixels, '
                                              'default 2250px',
                        type=int, default=DPI * WIDTH)
    parser.add_argument('--height', help='image height in pixels, '
                                               'default 1300px',
                        type=int, default=DPI * HEIGHT*.5)
    parser.add_argument('--dpi', help='image resolution in dpi, default 300dpi',
                        type=int, default=DPI)

    args = parser.parse_args()
    data = scipy.io.loadmat(args.mat, squeeze_me=True)

    parsed_data = dict()
    for key in ('AreaData', 'Silhouette', 'AreaNames'):
        try:
            parsed_data[key] = data[key]
        except KeyError:
            sys.exit('{} missing'.format(key))

    indexes = data.get('TickIndexes', None)
    tick_labels = data.get('TickLabels', None)
    colors = data.get('BarColors', None)

    f, (ax, ax2), legend = area_chart_silhouette(
        parsed_data['AreaData'], parsed_data['Silhouette'],
        parsed_data['AreaNames'], colors, args.xlabel, indexes, tick_labels,
        figsize=(args.width / float(args.dpi), args.height / float(args.dpi))
    )

    f.savefig(args.output, dpi=args.dpi, bbox_extra_artists=legend)