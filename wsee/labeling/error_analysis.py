import pandas as pd
import numpy as np
from wsee.data import explore


def get_false_positives(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int, label_of_interest: int):
    assert 'label' in labeled_df, 'Label column is missing, maybe call wsee.data.explore.add_labels'
    positives = labeled_df.iloc[lf_outputs[:, lf_index] == label_of_interest]
    false_positives = positives[positives['label'] != label_of_interest]
    return false_positives


def sample_fp(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int,
              label_of_interest: int, sample_size: int = 10):
    false_positives = get_false_positives(labeled_df, lf_outputs, lf_index, label_of_interest)
    return explore.sample_data(false_positives, sample_size=sample_size)


def get_abstained_instances(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int,
                            label_of_interest: int = -1):
    abstains = labeled_df.iloc[lf_outputs[:, lf_index] == -1]
    relevant_abstains = abstains[abstains['label'] == label_of_interest] if label_of_interest > -1 else abstains
    return relevant_abstains


def sample_abstained_instances(labeled_df: pd.DataFrame, lf_outputs: np.ndarray, lf_index: int,
                               label_of_interest: int = -1, sample_size: int = 10):
    abstains = get_abstained_instances(labeled_df, lf_outputs, lf_index, label_of_interest=label_of_interest)
    return explore.sample_data(abstains, sample_size=sample_size)
