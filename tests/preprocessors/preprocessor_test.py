import unittest

import pandas as pd
from wsee.preprocessors.preprocessors import get_mixed_ner


class TestMixedNer(unittest.TestCase):

    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/dataframes.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True)

    def test_mixed_ner(self):
        for index, gold_cand in self.pd_df.iterrows():
            expected_mixed_ner = gold_cand['mixed_ner']

            actual_cand = gold_cand.drop(columns=['mixed_ner'])

            actual_cand = get_mixed_ner(actual_cand)
            self.assertEqual(expected_mixed_ner, actual_cand['mixed_ner'])

    def test_mixed_ner_spans(self):
        for index, gold_cand in self.pd_df.iterrows():
            expected_mixed_ner_spans = gold_cand['mixed_ner_spans']
            # change from array spans to tuples to match with what is done in the preprocessor
            expected_mixed_ner_spans = [(span[0], span[1]) for span in expected_mixed_ner_spans]

            actual_cand = gold_cand.drop(columns=['mixed_ner_spans'])

            actual_cand = get_mixed_ner(actual_cand)
            self.assertEqual(expected_mixed_ner_spans, actual_cand['mixed_ner_spans'])


if __name__ == '__main__':
    unittest.main()
