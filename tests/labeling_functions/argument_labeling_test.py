import unittest

import pandas as pd
from wsee.data import pipeline, explore
from wsee.preprocessors import preprocessors
from wsee.labeling.event_argument_role_lfs import *


class TestMixedNer(unittest.TestCase):

    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/dataframes.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True)

    def test_mixed_ner(self):
        self.assertEqual(True, True)

    def test_somajo_separate_sentence(self):
        event_role_rows, event_role_rows_y = pipeline.build_event_role_examples(self.pd_df)
        labeled_rows = explore.add_labels(event_role_rows, event_role_rows_y)
        for idx, row in labeled_rows.iterrows():
            if row['argument']['text'] == 'ICE 1141' and row['trigger']['char_start'] == 535:
                self.assertEqual(lf_somajo_separate_sentence(row), no_arg)
        self.assertIsNotNone(event_role_rows)


if __name__ == '__main__':
    unittest.main()
