import os
import sys
import unittest
import numpy as np

from ..util import function as ft
from .. import ROOT_DIRECTORY


class TestWindow(unittest.TestCase):
    def setUp(self):
        self.test_sequence = list(range(5))

    def test_size_window_standard(self):
        sequences = [list(x) for x in ft.window(self.test_sequence, 2)]
        self.assertEqual(sequences, [[0, 1], [1, 2], [2, 3], [3, 4]])

        sequences = [list(x) for x in ft.window(self.test_sequence, 3)]
        self.assertEqual(sequences, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

    def test_too_small_sequence(self):
        sequences = [x for x in ft.window(self.test_sequence, 10)]
        self.assertEqual(sequences, [])

    def test_empty_sequence(self):
        sequences = [x for x in ft.window([])]
        self.assertEqual(sequences, [])


class TestMultiReplace(unittest.TestCase):

    def test_standard(self):
        replaced = ft.multi_replace(
            'aaa bbb ccc', {'aaa': 'ddd', 'ccc': 'eee'})
        self.assertEqual(replaced, 'ddd bbb eee')

    def test_long_short_replace(self):
        replaced = ft.multi_replace('aaa bbb ccc', {'aaa': 'ddd', 'aa': 'ee'})
        self.assertEqual(replaced, 'ddd bbb ccc')

    def test_no_matching(self):
        replaced = ft.multi_replace('aaa bbb ccc', {'ddd': 'eee'})
        self.assertEqual(replaced, 'aaa bbb ccc')

    def test_case_sensitive(self):
        replaced = ft.multi_replace(
            'AAA bbb ccc', {'aaa': 'eee', 'AAA': 'ddd'})
        self.assertEqual(replaced, 'ddd bbb ccc')

    def test_empty_replacement(self):
        replaced = ft.multi_replace('AAA bbb ccc', {})
        self.assertEqual(replaced, 'AAA bbb ccc')

    def test_empty_input(self):
        replaced = ft.multi_replace('', {'a': 'b'})
        self.assertEqual(replaced, '')
