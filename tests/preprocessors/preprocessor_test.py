import unittest

import pandas as pd
from wsee.preprocessors import preprocessors
from wsee.data import pipeline, explore


class TestMixedNer(unittest.TestCase):

    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/dataframes.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True)

    def test_mixed_ner(self):
        gold_cand = self.pd_df.iloc[0]
        expected_mixed_ner = gold_cand['mixed_ner']

        actual_cand = gold_cand.drop(columns=['mixed_ner'])

        actual_cand = preprocessors.pre_mixed_ner(actual_cand)
        # TODO: check trailing whitespace/ newline that is somehow introduced in preprocessor?
        self.assertEqual(expected_mixed_ner.strip(), actual_cand['mixed_ner'].strip())

        gold_cand = self.pd_df.iloc[1]
        expected_mixed_ner = gold_cand['mixed_ner']

        actual_cand = gold_cand.drop(columns=['mixed_ner'])

        actual_cand = preprocessors.pre_mixed_ner(actual_cand)
        # TODO: check trailing whitespace/ newline that is somehow introduced in preprocessor?
        self.assertEqual(expected_mixed_ner.strip(), actual_cand['mixed_ner'].strip())

    def test_mixed_ner_spans(self):
        gold_cand = self.pd_df.iloc[0]
        expected_mixed_ner_spans = gold_cand['mixed_ner_spans']
        # change from array spans to tuples to match with what is done in the preprocessor
        expected_mixed_ner_spans = [(span[0], span[1]) for span in expected_mixed_ner_spans]

        actual_cand = gold_cand.drop(columns=['mixed_ner_spans'])

        actual_cand = preprocessors.pre_mixed_ner(actual_cand)
        self.assertEqual(expected_mixed_ner_spans, actual_cand['mixed_ner_spans'])

        gold_cand = self.pd_df.iloc[1]
        expected_mixed_ner_spans = gold_cand['mixed_ner_spans']
        # change from array spans to tuples to match with what is done in the preprocessor
        expected_mixed_ner_spans = [(span[0], span[1]) for span in expected_mixed_ner_spans]

        actual_cand = gold_cand.drop(columns=['mixed_ner_spans'])

        actual_cand = preprocessors.pre_mixed_ner(actual_cand)
        self.assertEqual(expected_mixed_ner_spans, actual_cand['mixed_ner_spans'])

    def test_applypreprocessors(self):
        event_role_rows, event_role_rows_y = pipeline.build_event_role_examples(self.pd_df)
        labeled_rows = explore.add_labels(event_role_rows, event_role_rows_y)
        processed_rows = explore.apply_preprocessors(labeled_rows, pre=[preprocessors.pre_between_distance,
                                                                        preprocessors.pre_mixed_ner])
        self.assertIsNotNone(processed_rows)

    def test_getters(self):
        test = self.pd_df.iloc[3]
        test['trigger'] = {
            'id': 'c/1528f948-5baa-4da7-afbe-67c21edc4c58',
            'text': 'Stau', 'entity_type': 'trigger', 'start': 11, 'end': 12, 'char_start': 80, 'char_end': 84
        }
        test['argument'] = {
            'id': 'c/96e1c41b-63ad-4ca0-98fd-89ba7767b33c',
            'text': 'Swisttal', 'entity_type': 'location', 'start': 5, 'end': 6, 'char_start': 46, 'char_end': 54
        }
        argument_left_ner = preprocessors.get_windowed_left_ner(test['argument'], test['ner_tags'])
        self.assertEqual('O', argument_left_ner[-1])
        self.assertEqual('LOCATION_CITY', argument_left_ner[-2][2:])

        argument_left_tokens = preprocessors.get_windowed_left_tokens(test['argument'], test['tokens'])
        self.assertEqual('zwischen', argument_left_tokens[-1])


class TestComplexPreprocessors(unittest.TestCase):
    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/dataframes.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True)

    def test_somajo_preprocessor(self):
        event_type_rows, event_type_rows_y = pipeline.build_event_trigger_examples(self.pd_df)
        labeled_rows = explore.add_labels(event_type_rows, event_type_rows_y)
        processed_rows = explore.apply_preprocessors(labeled_rows, pre=[preprocessors.get_somajo_doc])
        self.assertIsNotNone(processed_rows)


if __name__ == '__main__':
    unittest.main()
