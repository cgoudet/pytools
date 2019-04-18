import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scstat
import seaborn as sns

logger = logging.getLogger(__name__)


def description(frame, outdir, target):
    """
    Save under a csv file the description of the dataset.

    :param DataFrame frame: data
    :param str outdir: saving directory
    :param str target: target variable with wich compute statistical test

    Add the dtypes to the describe method.
    """
    description = frame.describe().T
    description['dtype'] = frame.dtypes
    description['missing_values'] = frame.isnull().sum()
    description['fraction_missing'] = 100 * \
        description['missing_values'] / len(frame)

    description['nunique'] = frame.nunique()
    description.sort_index(inplace=True)

    binary_cols = [col for col in description[description['nunique']
                                              == 2].index.values if col != target]
    p_value = [chi2_indep(frame.loc[:, [col, target]]) for col in binary_cols]

    description['p_value_target'] = 1
    description.loc[binary_cols, 'p_value_target'] = p_value
    description.to_csv(outdir/'description.csv')

    return description


def univariate_plot(frame, outdir, target, is_clf):
    """
    Draw the distribution of the variable.

    :param DataFrame frame: dataset
    :param str outdir: saving directory
    :param str target: target variable column name
    :param bool is_clf: wether the problem is a classification

    Draw a violin plot for explicative variables and an histogram for
    the target variable.
    """

    corr = get_correlation(frame, outdir)

    for col in frame.columns:
        logger.info('plot', extra={'var': col})
        plt.figure()
        if col == target:
            sns.distplot(frame[col], kde=False)
        elif is_clf:
            sns.violinplot(x=target, y=col, data=frame, cut=0, sclae='area')
        else:
            sns.regplot(x=col, y=target, data=frame)

        if not (set([col, target]) - set(corr.columns)):
            plt.title('correlation : {:2.2f}'
                      .format(0
                              if pd.isna(corr.loc[col, target]) else
                              corr.loc[col, target]))

        plt.tight_layout()
        plt.savefig(str(outdir/'univariate_{}.png'.format(col)))
        plt.close()


def get_correlation(frame, outdir=None):
    """
    Draw a correlation heatmap and return the frame.

    :param DataFrame frame: data 
    :param str outdir: saving directory ( default=None=no saving)
    """

    corr = frame.corr()

    if outdir != None:
        corr.to_csv(outdir/'correlation.csv')
        plt.figure()
        sns.heatmap(corr, cmap=plt.cm.RdYlBu_r)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(str(outdir/'correlation.png'))
        plt.close()

    return corr


def chi2_indep(df):
    """
    Compute the p_value of variable independence.

    The dataset must contain only the two columns to compare.
    Duplicated columns are removed.
    """

    # clean duplicated columns
    df = df.loc[:, ~df.columns.duplicated()]
    if len(df.columns) != 2:
        raise RuntimeError("Incorrect number of columns")

    groupsizes = df.groupby(list(df.columns)).size()
    ctsum = groupsizes.unstack()
    chi2, p_value, dof, expected = scstat.chi2_contingency(ctsum.fillna(0))

    return p_value


def missing_values_table(df):
    """
    Return a table with the amount of missing values per column
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = df.isnull().sum()/len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'missing_values', 1: 'missing_values_perct'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            'missing_values_perct', ascending=False)

    # Print some summary information
    extra_logger = {'n_cols': df.shape[1], 'n_missing_cols': mis_val_table_ren_columns.shape[0], 'missing_cols': sorted(
        mis_val_table_ren_columns.index)}
    logger.info('missing_values', extra=extra_logger)

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
