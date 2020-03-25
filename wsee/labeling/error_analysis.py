import pandas as pd
import numpy as np
from wsee.data import explore


def get_false_positives(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int, label_of_interest: int):
    assert 'label' in labeled_df, 'Label column is missing, maybe call wsee.data.explore.add_labels'
    positives = labeled_df.iloc[lf_outputs[:, lf_index] == label_of_interest]
    false_positives = positives[positives['label'] != label_of_interest]
    return false_positives


def sample_fp(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int, label_of_interest: int):
    false_positives = get_false_positives(labeled_df, lf_outputs, lf_index, label_of_interest)
    return explore.sample_data(false_positives)


def trigger_text_counts_fp(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int, label_of_interest: int):
    false_positives = get_false_positives(labeled_df, lf_outputs, lf_index, label_of_interest)
    return false_positives['trigger_text'].value_counts()


def get_abstained_instances(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int):
    abstains = labeled_df.iloc[lf_outputs[:, lf_index] == -1]
    return abstains


def sample_abstained_instances(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int, label_of_interest: int):
    abstains = get_abstained_instances(labeled_df, lf_outputs, lf_index)
    return explore.sample_data(abstains[abstains['label'] == label_of_interest])


def trigger_text_counts_ai(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int, label_of_interest: int):
    abstains = get_abstained_instances(labeled_df, lf_outputs, lf_index)
    return abstains[abstains['label'] == label_of_interest]['trigger_text'].value_counts()
