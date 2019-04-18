import unittest
import pandas as pd
import numpy as np

from ..data_science import eda


class TestChi2(unittest.TestCase):

    def test_standard(self):
        # example taken from wikipedia : https://fr.wikipedia.org/wiki/Test_du_%CF%87%C2%B2
        entries = ([['A', '1']]*50 + [['B', '1']]*60 + [['A', '2']]*70
                   + [['B', '2']]*75 + [['A', '3']]*110 + [['B', '3']]*100
                   + [['A', '4']]*60 + [['B', '4']]*50)
        frame = pd.DataFrame(entries, columns=['col1', 'col2'])

        self.assertAlmostEqual(eda.chi2_indep(frame), 0.4892767468980648)
        self.assertAlmostEqual(eda.chi2_indep(
            frame.loc[:, ['col1', 'col2', 'col1']]), 0.4892767468980648)

        frame['test_col'] = 0
        with self.assertRaises(RuntimeError):
            eda.chi2_indep(frame)

        del frame['col1'], frame['col2']
        with self.assertRaises(RuntimeError):
            eda.chi2_indep(frame)

        with self.assertRaises(RuntimeError):
            eda.chi2_indep(frame.loc[:, ['test_col', 'test_col']])


class TestMissingValuesColumns(unittest.TestCase):
    def test_standard(self):
        df = pd.DataFrame([[0, 1, np.nan], [3, np.nan, 5], [
                          6, 7, np.nan]], columns=['col0', 'col1', 'col2'])

        output = eda.missing_values_table(df)

        expected = pd.DataFrame([[2, 2/3], [1, 1/3]], index=['col2', 'col1'],
                                columns=['missing_values', 'missing_values_perct'])

        pd.util.testing.assert_frame_equal(output, expected)
