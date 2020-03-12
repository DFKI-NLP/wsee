import unittest

import pandas as pd
from wsee.labeling.event_argument_role_lfs import *


class TestMixedNer(unittest.TestCase):

    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/dataframes.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True)

    def test_mixed_ner(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
