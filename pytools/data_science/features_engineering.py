import itertools
import functools
import math
import numpy as np
import pandas as pd


def stat_encoding(frame, category, agg=['mean', 'std'], prefix=''):
    """
    Apply aggregation to all non-category columns

    :param DataFrame frame: data
    :param list category: columns indentifying categories
    :param list agg: list of aggregating fucntions to apply
    :param str prefix: prefix for created columns name
    """

    # format agg to have expected form of column index
    if not isinstance(agg, list):
        agg = [agg]

    if not len(category):
        raise RuntimeError('Need category for stat_encoding')
    if not len(agg):
        raise RuntimeError('Need agg functions for stat_encoding')

    cat = frame.groupby(category).agg(agg)
    cat.columns = [prefix + '_'.join(col).strip()
                   for col in cat.columns.values]
    cat = cat.reset_index()

    return cat


def float_to_str(serie):
    """
    Transform a serie of float integer (with nan) into str
    """
    return serie.apply(lambda x: np.nan if pd.isna(x) else str(int(x))).astype(str)


def group_features(frame, category, agg,  prefix='', translate_names={}):
    """
    Merge computed group aggregation with frame

    :param DataFrame frame: dataset
    :param dict agg: aggregation functions
    :param list category: grouping columns

    :return DataFrame merged: input frame merged with aggregations

    """
    if not len(category):
        raise RuntimeError('Need category for stat_encoding')
    if not len(agg):
        raise RuntimeError('Need agg functions for stat_encoding')

    feat = frame.groupby(category).agg(agg)
    feat.columns = [x if isinstance(x, str) else '_'.join(x)
                    for x in feat.columns]
    translations = {col: col+'_' + (fct if isinstance(fct, str) else fct.__name__)
                    for col, fct in agg.items()
                    if not isinstance(fct, list)}

    feat.columns = [translations.get(c, c) for c in feat.columns]
    feat.columns = [prefix + col for col in feat.columns]
    feat.columns = [translate_names.get(c, c) for c in feat.columns]

    new_vars = pd.merge(frame.loc[:, category], feat, on=category)
    new_vars.drop(columns=category, inplace=True)

    return new_vars


def check_frame_columns(frame, columns):
    """
    Raise a RuntimeError if one of the columns is not present in frame.
    """
    for col in columns:
        if col not in frame.columns:
            raise RuntimeError('Missing column in frame : {}'.format(col))


def multi_date_features(frame, date_cols, levels):
    """ Add the date features to the dataframe

    Calls date_features for all columns
    """

    if isinstance(date_cols, str):
        date_cols = [date_cols]
    if not date_cols:
        raise RuntimeError('No date columns')

    if isinstance(levels, str):
        levels = [levels]
    if not levels:
        raise RuntimeError('No levels')

    part_date_feat = functools.partial(date_features, levels=levels)

    feats = [pd.DataFrame(
        frame[col].apply(part_date_feat).tolist(), columns=[col + '_' + ''.join(reversed(x))
                                                            for x in itertools.product(levels, ['', 'cos', 'sin'])], index=frame.index)
             for col in date_cols
             ]

    return pd.concat(feats, axis=1, sort=False)


def date_features(x, levels=['month', 'dayofweek', 'hourmin']):
    """Return a timestamp featurisation

    :param Timestamp x: the date to parse
    :param list levels: portions of the timestamp to featurise

    The accepted levels are : 'month', 'dayofweek', 'hourmin'

    For each level, in the decreasing order, the created variables will be :
    - value in its reference frame
    - cosine of its value with respect to the periodicity of its reference frame
    - sine of its value with respect to the periodicity of its reference frame
    """
    features = []
    for lev in levels:
        if 'month' == lev:
            val = x.month
            period = 12
        elif 'dayofweek' == lev:
            val = x.dayofweek
            period = 7
        elif 'hourmin' == lev:
            val = x.minute + x.hour*60
            period = 24 * 60
        elif 'weekhour' == lev:
            val = x.hour + x.dayofweek*24
            period = 24 * 7
        elif 'hour' == lev:
            val = x.hour
            period = 24
        elif 'dayofyear' == lev:
            val = x.dayofyear
            period = 366
        else:
            continue

        pulse = 2*math.pi / period
        features.extend([val, math.cos(val*pulse), math.sin(val*pulse)])

    return features


def single_val_cols(df):
    """
    Returns the columns which contains a single value.
    """

    cols_unique_vals = df.nunique(axis=0)
    cols = cols_unique_vals[cols_unique_vals <= 1].index.values

    return cols


def group_cumulative(df, group, values):
    """
    Cumulative sum within each category.

    It is assumed that the dataset is already sorted by group features, and sorted within each category.
    """

    if not isinstance(group, list):
        group = [group]
    if not isinstance(values, list):
        values = [values]
    all_cols = group + values
    df = df.loc[:, all_cols].copy()
    df.loc[:, values] = df.loc[:, values].cumsum()

    agg_actions = {col: min for col in values}
    shifted = df.copy()
    shifted.loc[:, values] = shifted.loc[:, values].shift(1, fill_value=0)
    min_vals = shifted.groupby(group).agg(agg_actions).fillna(0)
    min_vals.columns = 'min_'+min_vals.columns

    df = pd.merge(df, min_vals, on=group)
    df.loc[:, values] -= df.loc[:, ['min_'+c for c in values]].values

    dropped_columns = set(df.columns) - set(values)
    df.drop(columns=dropped_columns, inplace=True)

    return df
