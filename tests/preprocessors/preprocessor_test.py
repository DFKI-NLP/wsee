import unittest

import pandas as pd
from wsee.preprocessors import preprocessors
from wsee.data import pipeline, explore


class TestMixedNer(unittest.TestCase):

    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/dataframes.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True)

    def test_mixed_ner(self):
        for index, gold_cand in self.pd_df.iterrows():
            expected_mixed_ner = gold_cand['mixed_ner']

            actual_cand = gold_cand.drop(columns=['mixed_ner'])

            actual_cand = preprocessors.get_mixed_ner(actual_cand)
            # TODO: check trailing whitespace/ newline that is somehow introduced in preprocessor?
            self.assertEqual(expected_mixed_ner.strip(), actual_cand['mixed_ner'].strip())

    def test_mixed_ner_spans(self):
        for index, gold_cand in self.pd_df.iterrows():
            expected_mixed_ner_spans = gold_cand['mixed_ner_spans']
            # change from array spans to tuples to match with what is done in the preprocessor
            expected_mixed_ner_spans = [(span[0], span[1]) for span in expected_mixed_ner_spans]

            actual_cand = gold_cand.drop(columns=['mixed_ner_spans'])

            actual_cand = preprocessors.get_mixed_ner(actual_cand)
            self.assertEqual(expected_mixed_ner_spans, actual_cand['mixed_ner_spans'])

    def test_applypreprocessors(self):
        event_type_rows, event_type_rows_y = pipeline.build_event_trigger_examples(self.pd_df)
        labeled_rows = explore.add_labels(event_type_rows, event_type_rows_y)
        processed_rows = explore.apply_preprocessors(labeled_rows, pre=[preprocessors.pre_between_distance,
                                                                        preprocessors.pre_mixed_ner])
        self.assertIsNotNone(processed_rows)


class TestComplexPreprocessors(unittest.TestCase):
    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/dataframes.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True)

    def test_stanford_preprocessor(self):
        event_type_rows, event_type_rows_y = pipeline.build_event_trigger_examples(self.pd_df)
        labeled_rows = explore.add_labels(event_type_rows, event_type_rows_y)
        processed_rows = explore.apply_preprocessors(labeled_rows, pre=[preprocessors.pre_stanford_doc])
        self.assertIsNotNone(processed_rows)

    def test_spacy_preprocessor(self):
        event_type_rows, event_type_rows_y = pipeline.build_event_trigger_examples(self.pd_df)
        labeled_rows = explore.add_labels(event_type_rows, event_type_rows_y)
        processed_rows = explore.apply_preprocessors(labeled_rows, pre=[preprocessors.pre_spacy_doc])
        self.assertIsNotNone(processed_rows)

    def test_somajo_preprocessor(self):
        event_type_rows, event_type_rows_y = pipeline.build_event_trigger_examples(self.pd_df)
        labeled_rows = explore.add_labels(event_type_rows, event_type_rows_y)
        processed_rows = explore.apply_preprocessors(labeled_rows, pre=[preprocessors.get_somajo_doc])
        self.assertIsNotNone(processed_rows)


if __name__ == '__main__':
    unittest.main()
