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


def add_event_arg_roles(dataframe: pd.DataFrame):
    """
    If the dataframe contains gold labels for the events, add a column containing the events and their arguments in the
    form of (trigger text, event type label, argument text, event argument label).
    :param dataframe: DataFrame.
    :return: DataFrame with event types column.
    """
    if 'event_roles' in dataframe:
        dataframe = dataframe.apply(get_event_arg_roles, axis=1)
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


def sample_data(x: pd.DataFrame, sample_size: int = 10, columns: List[str] = None):
    if columns is not None:
        cols = [col for col in columns if col in x]
        if len(x) < sample_size:
            return x[cols]
        else:
            return x.sample(sample_size)[cols]
    elif len(x) < sample_size:
        return x
    else:
        return x.sample(sample_size)