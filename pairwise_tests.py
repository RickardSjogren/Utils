#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains a function to calculate parwise-test statistics
for multiclass data-tables.

Can be run as command-line script.
"""
import numpy as np
import pandas as pd
from scipy import stats


def pairwise_test(first_data, second_data, test, nan_action='skip',
                  *args, **kwargs):
    """ Test columns of first_data against columns of second_data.

    Additional arguments are passed to test-function used.
    Signature of `test`is assumed to be::
    
        test(x: array_like, y: array_like, *args, **kwargs) -> float, float
    
    Parameters
    ----------
    first_data, second_data : array_like
        Datasets with n columns, columns must match.
    test : callable
        Which test to use.
    nan_action : str
        'skip' | 'omit_nans'
    *args
        Additional arguments passed to `test`
    **kwargs
        Addition keyword arguments passed to `test`
        
    Returns
    -------
    statistics : array_like
        One-dimensional array with test-statistics (length n)
    pvalues : array_like
        One-dimensional array with p-values (lenght n)
    """
    if not nan_action in ('skip', 'omit_nan'):
        raise ValueError('unknown nan_action: {0}'.format(nan_action))
    
    if type(first_data) == type(second_data) == type(pd.DataFrame()):
        if not (set(first_data.columns) == set(second_data.columns)):
            raise ValueError("Datasets don't match")
        
    else:
        # If not dataframes, assume that columns match if number of columns
        # match.
        if not (first_data.shape[1] == second_data.shape[1]):
            raise ValueError("Datasets don't match")
        first_data = pd.DataFrame(first_data)
        second_data = pd.DataFrame(second_data)
            
    statistics = np.zeros((first_data.shape[1], ))
    pvalues = np.zeros((first_data.shape[1], ))
    for i, column in enumerate(first_data.columns):
        raw_x = first_data[column].values
        raw_y = second_data[column].values
        
        if nan_action == 'omit_nans':
            # Omit positions which are NaN in any vector.
            missing = np.logical_or(np.isnan(raw_x), np.isnan(raw_y))
            x = raw_x[~missing]
            y = raw_y[~missing]
            
        elif np.logical_or(np.isnan(raw_x).any(), np.isnan(raw_y).any()):
            # If skipping NaN:s, set statistic and p-value to NaN if any
            # vector contains any NaN.
            statistics[i] = pvalues[i] = np.nan
            continue
        else:
            x = raw_x
            y = raw_y            
            
        statistic, p = test(x, y, *args, **kwargs)
        statistics[i] = statistic
        pvalues[i] = p
        
    return statistics, pvalues
    

if __name__ == '__main__':
    import argparse
    
    allowed_tests = ('student-t', 'mann-whitney-u', )    
    
    parser = argparse.ArgumentParser(description=(
        'This program performs column-wise statistical tests against the '
        'null hypothesis that the samples are drawn from the same '
        'population.'
    ))

    def check_csv(fname):
        if not fname.endswith('.csv'):
            raise argparse.ArgumentTypeError('datasets must be csv-file')
        else:
            return fname
    
    parser.add_argument('datasets', metavar='Data', nargs=2, type=check_csv,
                        help='csv-files to perform column-wise testing at')
                        
    test_help = (
        'which test to use, available: {0}. '
        'Default: Student-T'
    ).format(', '.join(test.title() for test in allowed_tests))
    parser.add_argument('--test', help=test_help,
                        choices=allowed_tests, default='student-t')
    o_help = 'output file. Defaults to: <data1>_<data2>_<test>.csv'
    parser.add_argument('-o', '--output', type=str, 
                        help=o_help)
                        
    group = parser.add_mutually_exclusive_group()
    omit_help = (
        'if set, calculate statistic using non-NaN-values '
        'for those columns containing any missing values.'
    )
    group.add_argument('--omit_nans', dest='nan_action', action='store_const',
                       const='omit_nan',
                       help=omit_help)
                       
    skip_help = (
        'if set, skip calculating statistic in those cases '
        'where a column contains missing values.' 
    )
    group.add_argument('--skip', dest='nan_action', action='store_const', 
                       const='skip', help=skip_help)
    parser.set_defaults(nan_action='skip', test='student-t')
    
    args = parser.parse_args()
    
    if args.test == 'student-t':
        test = stats.ttest_ind
    elif args.test == 'mann-whitney-u':
        test = stats.mannwhitneyu
        
    first_data = pd.DataFrame.from_csv(args.datasets[0])
    second_data = pd.DataFrame.from_csv(args.datasets[1])
    
    statistics, pvalues = pairwise_test(first_data, second_data,
                                        test, args.nan_action)
    results = pd.DataFrame(np.column_stack((statistics, pvalues)), 
                           index=first_data.columns, columns=['statistic', 'p'])

    if args.output:
        out = args.output
    else:
        out = '{0}_{1}_{2}.csv'.format(
            args.datasets[0].replace('.csv', ''),
            args.datasets[1].replace('.csv', ''),
            args.test.replace('-', '_')
        )
                               
    results.to_csv(out if out.endswith('.csv') else out + '.csv')
                                        
