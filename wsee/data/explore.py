import pandas as pd
from tqdm import tqdm
from snorkel.preprocess import BasePreprocessor
from wsee.preprocessors.preprocessors import *


def add_labels(dataframe: pd.DataFrame, labels):
    dataframe['label'] = list(labels)
    return dataframe


def add_event_types(dataframe: pd.DataFrame):
    """
    If the dataframe contains gold labels for the events, add a column containing the sequence of events in the form of
    (trigger text, event type label).
    :param dataframe: DataFrame.
    :return: DataFrame with event types column.
    """
    if 'event_triggers' in dataframe:
        dataframe = dataframe.apply(get_event_types, axis=1)
    return dataframe


def apply_preprocessors(x: pd.DataFrame, pre: List[BasePreprocessor]):
    """
    Applies preprocessors to dataframe. (Essentially _preprocess_data_point function
    from Snorkel's LabelingFunction class)
    :param x: DataFrame.
    :param pre: Preprocessors to run on data points.
    :return:
    """
    for pre_func in tqdm(pre):
        x = x.apply(pre_func, axis=1)
        if x is None:
            raise ValueError("Preprocessor should not return None")
    return x


def sample_data(x: pd.DataFrame, sample_size: int = 10,
                columns: List[str] = (
                        'trigger_left_tokens', 'trigger_text', 'trigger_right_tokens', 'entity_type_freqs',
                        'mixed_ner', 'label', 'event_types')):
    if len(x) < sample_size:
        return x[list(columns)]
    else:
        return x.sample(sample_size)[list(columns)]
