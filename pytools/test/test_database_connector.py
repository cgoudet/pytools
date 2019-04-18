import json
import os
import sys
import unittest

from .. import ROOT_DIRECTORY
from ..database import connector


class TestGetConfig(unittest.TestCase):
    def setUp(self):
        self.test_data_path = ROOT_DIRECTORY/'data'/'test'

    def test_unknown_file(self):
        self.assertRaises(FileNotFoundError, connector.get_config,
                          'useless_id', 'wrong_config_directory')

    def test_unknown_base(self):
        self.assertRaises(FileNotFoundError, connector.get_config,
                          'wrong_id', self.test_data_path)

    def test_wrong_json_format(self):
        self.assertRaises(json.decoder.JSONDecodeError,
                          connector.get_config, 'wrong_json_format', self.test_data_path)

    def test_ok(self):
        label0_data = connector.get_config(
            'valid_json_format', self.test_data_path)
        self.assertEqual(len(label0_data), 2)
        self.assertEqual(label0_data['label0'], 'value0')
