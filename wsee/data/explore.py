import pandas as pd
from snorkel.preprocess import BasePreprocessor
from wsee.preprocessors.preprocessors import *


def add_labels(dataframe: pd.DataFrame, labels):
    dataframe['label'] = list(labels)
    return dataframe


def apply_preprocessors(x: pd.DataFrame, pre: List[BasePreprocessor]):
    """
    Applies preprocessors to dataframe. (Essentially _preprocess_data_point function
    from Snorkel's LabelingFunction class)
    :param x: DataFrame.
    :param pre: Preprocessors to run on data points.
    :return:
    """
    for pre_func in pre:
        x = x.apply(pre_func, axis=1)
        if x is None:
            raise ValueError("Preprocessor should not return None")
    return x
