import unittest

import pandas as pd
from wsee.preprocessors import preprocessors


class TestSentenceSplitter(unittest.TestCase):

    def setUp(self):
        dataframes_path = '/Users/phuc/develop/python/wsee/tests/fixtures/daystream_sample.jsonl'
        self.pd_df: pd.DataFrame = pd.read_json(dataframes_path, lines=True, encoding='utf8')

    def test_sentence_splitting(self):
        preprocessors.load_somajo_model()
        nlp_somajo = preprocessors.nlp_somajo
        for idx, row in self.pd_df.iterrows():
            text = row['text']
            somajo_doc = list(nlp_somajo.tokenize_text([text]))
            somajo_sentences = preprocessors.get_somajo_doc_sentences(somajo_doc, text)
            for sentence in somajo_sentences:
                self.assertEqual(sentence['text'], text[sentence['char_start']:sentence['char_end']])


if __name__ == '__main__':
    unittest.main()
