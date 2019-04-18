import unittest

import numpy as np
import pandas as pd

from ..data_science import features_engineering as feateng


class TestMeanEncoding(unittest.TestCase):

    def setUp(self):
        self.frame = pd.DataFrame([[1, 2, 3], [1, 1, 3], [1, 4, 5], [1, 5, 5], [2, 6, 1], [
                                  2, 2, 1], [2, 8, 3], [2, 12, 3]], columns=['col0', 'col1', 'col2'])

    def test_2cat_2agg(self):
        output = feateng.stat_encoding(self.frame, category=['col0', 'col2'])
        expected = pd.DataFrame([[1, 3, 1.5, 0.707107], [1, 5, 4.5, 0.707107], [2, 1, 4, 2.828427], [
                                2, 3, 10, 2.828427]], columns=['col0', 'col2', 'col1_mean', 'col1_std'])

        pd.util.testing.assert_frame_equal(output, expected)

    def test_1cat_1agg(self):
        output = feateng.stat_encoding(
            self.frame.loc[:, ['col0', 'col1']], category='col0', agg='mean')
        expected = pd.DataFrame([[1, 3], [2, 7]], columns=[
                                'col0', 'col1_mean'])
        pd.util.testing.assert_frame_equal(output, expected)

    def test_multi_target_col(self):
        output = feateng.stat_encoding(self.frame, category='col0', agg='mean')
        expected = pd.DataFrame([[1, 3, 4], [2, 7, 2]], columns=[
                                'col0', 'col1_mean', 'col2_mean'])
        pd.util.testing.assert_frame_equal(output, expected)

    def test_custom_agg(self):
        output = feateng.stat_encoding(
            self.frame.loc[:, ['col0', 'col1']], category='col0', agg=lambda x: len(np.unique(x)))
        expected = pd.DataFrame([[1, 4], [2, 4]], columns=[
                                'col0', 'col1_<lambda>'])
        pd.util.testing.assert_frame_equal(output, expected)

    def test_errors(self):

        with self.assertRaises(RuntimeError):
            feateng.stat_encoding(self.frame, category=[])

        with self.assertRaises(RuntimeError):
            feateng.stat_encoding(self.frame, category=['col1'], agg=[])


class TestAddGroupFeatures(unittest.TestCase):

    def test_standard(self):
        frame = pd.DataFrame([[1, 2, 3]
                              , [1, 1, 4]
                              , [2, 6, 1]
                              , [2, 2, 1]]
                             , columns=['cat', 'col0', 'col1'])
        agg = {'col0': [sum, 'mean'], 'col1': [np.mean]}

        added_frame = feateng.group_features( frame, ['cat'], agg)

        expected = pd.DataFrame( [[3.0, 1.5, 3.5]
                                  , [3, 1.5, 3.5]
                                  , [8, 4, 1]
                                  , [8, 4, 1]]
                                  , columns=['col0_sum', 'col0_mean', 'col1_mean'])

        self.assertEqual( sorted(expected.columns), sorted(added_frame.columns))
        #pd.util.testing.assert_frame_equal(added_frame.loc[:,expected.columns], expected)

    def test_no_list(self):
        frame = pd.DataFrame([[1, 1, 1], [1, 2, 3], [2, 4, 4]]
                             , columns=['cat', 'col0', 'col1'])
        agg = {'col0': sum}

        added_frame = feateng.group_features( frame,['cat'], agg)

        expected = pd.DataFrame([[3], [3], [4]]
                                , columns=['col0_sum'])

        pd.util.testing.assert_frame_equal( added_frame, expected )

    def test_no_categ(self):
        frame = pd.DataFrame([[1, 1], [1, 2]]
                             , columns=['cat', 'col0'])
        agg = {'col0': 'mean'}

        with self.assertRaises(RuntimeError):
            feateng.group_features(frame, [], agg)

    def test_no_agg(self):
        frame = pd.DataFrame([[1, 1], [1, 2]]
                             , columns=['cat', 'col0'])

        with self.assertRaises(RuntimeError):
            feateng.group_features(frame, ['cat'], {})

    def test_prefix(self):

        frame = pd.DataFrame([[1, 1, 1]
                              , [1, 2, 3]
                              , [2, 4, 4]]
                              , columns=['cat', 'col0', 'col1']
                              , dtype=np.int8)
        agg = {'col0': sum}

        added_frame = feateng.group_features(
            frame, ['cat'], agg, prefix='prefix_')

        expected = pd.DataFrame( [[3], [3], [4]]
                                , columns=['prefix_col0_sum']
                                , dtype=np.int8)

        pd.util.testing.assert_frame_equal(
            added_frame, expected)

    def test_translate_name(self):
        frame = pd.DataFrame( [[1, 1, 1]
                           , [1, 2, 3]
                           , [2, 4, 4]]
                           , columns=['cat', 'col0', 'col1']
                           , dtype=np.int8)
        agg = {'col0': sum}
        translate_names = {'col0_sum': 'sum_col0'}

        added_frame = feateng.group_features(
            frame, 'cat', agg, translate_names=translate_names)

        expected = pd.DataFrame( [[3], [3], [4]]
                                , columns=['sum_col0']
                                , dtype=np.int8)

        pd.util.testing.assert_frame_equal(added_frame, expected)

class TestCheckFrameColumns(unittest.TestCase):
    def test_all_ok(self):
        frame = pd.DataFrame(np.arange(4).reshape(
            (2, 2)), columns=['col1', 'col2'])

        feateng.check_frame_columns(frame, ['col1', 'col2'])

    def test_wrong_col(self):
        frame = pd.DataFrame(np.arange(4).reshape(
            (2, 2)), columns=['col1', 'col2'])

        with self.assertRaises(RuntimeError):
            feateng.check_frame_columns(frame, ['col1', 'wrong_col'])


class TestDateFeatures(unittest.TestCase):

    def setUp(self):
        self.ts = pd.Timestamp('2018-12-18 07:10:10')

    def test_empty_levels(self):
        self.assertEqual([], feateng.date_features(self.ts, []))
        self.assertEqual([], feateng.date_features(self.ts, ['unknown']))

    def test_hourmin(self):
        features = feateng.date_features(self.ts, ['hourmin'])

        # ( 7h=, cos( 2pi/24*7), sin(2pi*(7*60+10)/24/60) )
        expected = [430, -0.30070579950427295, 0.9537169507482269]
        np.testing.assert_almost_equal(features, expected, 1e-5)

    def test_multi_month(self):
        features = feateng.date_features(self.ts, ['month'])
        expected = [12, 1, 0]  # december over 12 months
        np.testing.assert_almost_equal(features, expected, 1e-5)

    def test_multi_dayofyear_dayofweek(self):

        features = feateng.date_features(self.ts, ['dayofyear', 'dayofweek'])
        expected = [352, 0.9712569994658311, -0.23803327706148678,
                    1, 0.6234898018587336, 0.7818314824680298]  # tuesday

        np.testing.assert_almost_equal(features, expected, 1e-5)

    def test_multi_weekhour(self):
        features = feateng.date_features(self.ts, ['weekhour'])
        expected = [31, 0.3998920243197411, 0.9165622558699761]
        np.testing.assert_almost_equal(features, expected, 1e-5)

    def test_multi_weekhour(self):
        features = feateng.date_features(self.ts, ['hour'])
        expected = [7, -0.25881904510252063, 0.9659258262890683]
        np.testing.assert_almost_equal(features, expected, 1e-5)


class TestMultiDateFeatures(unittest.TestCase):

    def test_single_col(self):

        df = pd.DataFrame([[pd.Timestamp('2018-12-18 07:10:10')]
                           , [pd.Timestamp('2018-12-18 07:10:10')]]
                          , columns=['col0'])

        out_df = feateng.multi_date_features(
            df, date_cols=['col0'], levels=['hour'])

        hour = 7
        coshour = -0.25881904510252063
        sinhour = 0.9659258262890683
        expected = pd.DataFrame( [[hour, coshour, sinhour],
                                [hour, coshour, sinhour]]
                                , columns=['col0_hour', 'col0_coshour', 'col0_sinhour'])

        pd.util.testing.assert_frame_equal(out_df, expected)

    def test_multi_levels_col(self):
        df = pd.DataFrame([[pd.Timestamp('2018-12-18 07:10:10'), pd.Timestamp('2018-12-18 07:10:10')]
                           , [pd.Timestamp('2018-12-18 07:10:10'), pd.Timestamp('2018-12-18 07:10:10')]]
                          , columns=['col0', 'col1'])

        out_df = feateng.multi_date_features(df, date_cols=['col0', 'col1'], levels=['hour', 'dayofweek'])

        hour = 7
        coshour = -0.25881904510252063
        sinhour = 0.9659258262890683

        dayofweek = 1
        cosdayofweek = 0.6234898018587336
        sindayofweek = 0.7818314824680298
        expected = pd.DataFrame( [[hour, coshour, sinhour, dayofweek, cosdayofweek, sindayofweek, hour, coshour, sinhour, dayofweek, cosdayofweek, sindayofweek]
                                  , [hour, coshour, sinhour, dayofweek, cosdayofweek, sindayofweek, hour, coshour, sinhour, dayofweek, cosdayofweek, sindayofweek]]
                                , columns=['col0_hour', 'col0_coshour', 'col0_sinhour', 'col0_dayofweek', 'col0_cosdayofweek', 'col0_sindayofweek'
                                           ,'col1_hour', 'col1_coshour', 'col1_sinhour', 'col1_dayofweek', 'col1_cosdayofweek', 'col1_sindayofweek'])

        pd.util.testing.assert_frame_equal(expected, out_df)

    def test_no_list(self):

        df = pd.DataFrame([[pd.Timestamp('2018-12-18 07:10:10')]
                           , [pd.Timestamp('2018-12-18 07:10:10')]]
                          , columns=['col0'])

        out_df = feateng.multi_date_features(
            df, date_cols='col0', levels='hour')

        hour = 7
        coshour = -0.25881904510252063
        sinhour = 0.9659258262890683
        expected = pd.DataFrame( [[hour, coshour, sinhour],
                                [hour, coshour, sinhour]]
                                , columns=['col0_hour', 'col0_coshour', 'col0_sinhour'])

        pd.util.testing.assert_frame_equal(out_df, expected)

    def test_empty(self):

        df = pd.DataFrame([[pd.Timestamp('2018-12-18 07:10:10')]
                           , [pd.Timestamp('2018-12-18 07:10:10')]]
                          , columns=['col0'])

        with self.assertRaises(RuntimeError) :
            feateng.multi_date_features( df, date_cols=[], levels='hour')

        with self.assertRaises(RuntimeError) :
            feateng.multi_date_features( df, date_cols=['col0'], levels=[])

class TestGetSingleValCols(unittest.TestCase):
    def setUp(self):
        pass

    def test_nosingle(self):

        df = pd.DataFrame(np.arange(4).reshape(
            (2, 2)), columns=['col0', 'col1'])

        cols = feateng.single_val_cols(df)

        self.assertEqual(len(cols), 0)

    def test_single_and_zero_values(self):
        df = pd.DataFrame([[1, np.nan, np.nan, 1], [1, 1, np.nan, 2]], columns=[
                          'col0', 'col1', 'col2', 'col3'])

        cols = feateng.single_val_cols(df)

        self.assertEqual(list(cols), ['col0', 'col1', 'col2'])

    def test_empty_frame(self):

        df = pd.DataFrame([])

        cols = feateng.single_val_cols(df)

        self.assertEqual(len(cols), 0)

class TestGroupCumulative(unittest.TestCase):
    def setUp(self):
        pass
    def test_standard(self):
        inputs = pd.DataFrame([[0, 0, 0.0, 1.0]
                               , [0, 0, 1, 0]
                               , [0, 0, 0, 1]
                               , [0, 1, 1, 0]]
                              , columns=['group0', 'group1', 'val0', 'val1'])
        expected = pd.DataFrame([[0.0, 1.0]
                                , [1, 1]
                                , [1, 2]
                                , [1, 0]]
                                , columns=['val0', 'val1'])

        output = feateng.group_cumulative(inputs, group=['group0', 'group1'], values=['val0', 'val1'])

        pd.util.testing.assert_frame_equal(expected, output)

    def test_nan_values(self):
        inputs = pd.DataFrame([[  0, np.nan]
                               , [0, np.nan]
                               , [0, np.nan]
                               , [1, np.nan]
                               , [1, -10]
                               , [2, 3.2]
                               , [2, np.nan]
                               , [2, -6.3]
                               , [3, 14]
                               , [3, np.nan]
                               , [3, 8]]
                              , columns=['group0', 'val0'])
        expected = pd.DataFrame(np.array([np.nan, np.nan, np.nan, np.nan, -10, 3.2, np.nan, -3.1, 14, np.nan, 22]).reshape((-1, 1))
                                , columns=['val0'])

        output = feateng.group_cumulative(inputs, group='group0', values='val0')

        pd.util.testing.assert_frame_equal(expected, output)
