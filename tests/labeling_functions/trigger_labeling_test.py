import unittest

import pandas as pd
from wsee.labeling.event_trigger_lfs import *


class TestKeyword(unittest.TestCase):

    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/dataframes.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True)

    def test_keyword(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
