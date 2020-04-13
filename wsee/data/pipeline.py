import os
from pathlib import Path

import pandas as pd
from snorkel.labeling import LabelModel, PandasLFApplier
from tqdm import tqdm

from wsee.labeling.event_argument_role_lfs import *
from wsee.labeling.event_trigger_lfs import *


def load_data(path, use_build_defaults=True):
    """
    Loads corpus data from specified path.
    :param path: Path to corpus directory.
    :param use_build_defaults: Whether to use data with defaults (trigger-entity pairs without
    annotation in the data are assigned a negative label) or only the data where only the original
    avro annotation was used.
    :return: output_dict containing train, dev, test, daystream data.
    """
    input_path = Path(path)
    assert input_path.exists(), 'Input not found: %s'.format(path)

    output_dict = {}
    for split in ['train', 'dev', 'test']:
        if use_build_defaults:
            sd_path = input_path.joinpath(split, f'{split}_with_events_and_defaults.jsonl')
        else:
            sd_path = input_path.joinpath(split, f'{split}_with_events.jsonl')
        assert os.path.exists(sd_path)
        sd_data = pd.read_json(sd_path, lines=True)
        output_dict[split] = sd_data

    daystream_path = os.path.join(input_path, 'daystream.jsonl')
    assert os.path.exists(daystream_path)
    daystream = pd.read_json(daystream_path, lines=True)
    output_dict['daystream'] = daystream

    return output_dict


def build_event_trigger_examples(dataframe):
    """
    Takes a dataframe containing one document per row with all its annotations
    (event triggers are of interest here) and creates one row for each event trigger.
    :param dataframe: Annotated documents.
    :return: DataFrame containing event trigger examples and NumPy array containing labels.
    """
    event_trigger_rows = []
    event_trigger_rows_y = []

    event_count = 0

    print(f"DataFrame has {len(dataframe.index)} rows")
    for index, row in tqdm(dataframe.iterrows()):
        for event_trigger in row.event_triggers:
            augmented_row = utils.get_deep_copy(row)
            augmented_row['trigger_id'] = event_trigger['id']
            event_trigger_rows.append(augmented_row)
            event_type_num = np.asarray(event_trigger['event_type_probs']).argmax()
            event_trigger_rows_y.append(event_type_num)
            if event_type_num != 7:
                event_count += 1

    print("Number of events:", event_count)
    event_trigger_rows = pd.DataFrame(event_trigger_rows)
    event_trigger_rows_y = np.asarray(event_trigger_rows_y)
    return event_trigger_rows, event_trigger_rows_y


def build_event_role_examples(dataframe):
    """
    Takes a dataframe containing one document per row with all its annotations
    (event roles are of interest here) and creates one row for each trigger-entity
    (event role) pair.
    :param dataframe: Annotated documents.
    :return: DataFrame containing event role examples and NumPy array containing labels.
    """
    event_role_rows_list = []
    event_role_rows_y = []

    event_count = 0

    for index, row in tqdm(dataframe.iterrows()):
        for event_role in row.event_roles:
            augmented_row = utils.get_deep_copy(row)
            augmented_row['trigger_id'] = event_role['trigger']
            augmented_row['argument_id'] = event_role['argument']
            event_role_rows_list.append(augmented_row)
            event_role_num = np.asarray(event_role['event_argument_probs']).argmax()
            event_role_rows_y.append(event_role_num)
            if event_role_num != 10:
                event_count += 1

    print("Number of event roles:", event_count)
    event_role_rows = pd.DataFrame(event_role_rows_list).reset_index(drop=True)
    event_role_rows_y = np.asarray(event_role_rows_y)

    return event_role_rows, event_role_rows_y


def build_labeled_event_trigger(x):
    """
    Builds event_trigger for example.
    :param x: DataFrame row containing one event trigger example.
    :return: DataFrame row with filled event_triggers column.
    """
    event_trigger = {
        'id': x.trigger_id,
        'event_type_probs': x.event_type_probs
    }
    x['event_triggers'] = [event_trigger]
    return x


def merge_event_trigger_examples(event_trigger_rows, event_trigger_probs):
    """
    Merges event trigger examples belonging to the same document.
    :param event_trigger_rows: DataFrame containing the event trigger examples.
    :param event_trigger_probs: NumPy array containing the event trigger class probabilities.
    :return: DataFrame containing one document per row.
    """
    # add event_trigger_probs to dataframe as additional column
    event_trigger_rows['event_type_probs'] = list(event_trigger_probs)
    event_trigger_rows = event_trigger_rows.apply(build_labeled_event_trigger, axis=1)
    aggregation_functions = {
        'event_triggers': 'sum',  # expects list of one trigger per row
    }
    return event_trigger_rows.groupby('id').agg(aggregation_functions)


def build_labeled_event_role(x):
    """
    Builds event_role for example.
    :param x: DataFrame row containing one event role example.
    :return: DataFrame row with filled event_roles column.
    """
    event_role = {
        'trigger': x.trigger_id,
        'argument': x.argument_id,
        'event_argument_probs': x.event_argument_probs
    }
    x['event_roles'] = [event_role]
    return x


def merge_event_role_examples(event_role_rows: pd.DataFrame, event_argument_probs) -> pd.DataFrame:
    """
    Merges event role examples belonging to the same document.
    :param event_role_rows: DataFrame containing the event role examples.
    :param event_argument_probs: NumPy array containing the event role class probabilities.
    :return: DataFrame containing one document per row.
    """
    # add event_trigger_probs to dataframe as additional column
    # TODO fix: A value is trying to be set on a copy of a slice from a DataFrame
    event_role_rows_copy = event_role_rows.copy()
    event_role_rows_copy['event_argument_probs'] = list(event_argument_probs)
    event_role_rows_copy = event_role_rows_copy.apply(build_labeled_event_role, axis=1)
    aggregation_functions = {
        'event_roles': 'sum'  # expects list of one event role per row
    }
    return event_role_rows_copy.groupby('id').agg(aggregation_functions)


def get_trigger_probs(l_train: pd.DataFrame, lfs: Optional[List[labeling_function]] = None,
                      label_model: LabelModel = None):
    """
    Takes "raw" data frame, builds trigger examples, (trains LabelModel), calculates event_trigger_probs
    and returns merged trigger examples with event_trigger_probs.
    :param l_train:
    :param lfs:
    :param label_model:
    :return:
    """
    event_trigger_examples, _ = build_event_trigger_examples(l_train)
    if lfs is None:
        lfs = [
            lf_accident_cat,
            lf_canceledroute_cat,
            lf_canceledstop_cat,
            lf_delay_cat,
            lf_obstruction_cat,
            lf_railreplacementservice_cat,
            lf_trafficjam_cat
        ]
    applier = PandasLFApplier(lfs)
    df_train = applier.apply(event_trigger_examples)
    if label_model is None:
        label_model = LabelModel(cardinality=8, verbose=True)
        label_model.fit(L_train=df_train, n_epochs=500, log_freq=100, seed=123)
    event_trigger_probs = label_model.predict_proba(df_train)
    return merge_event_trigger_examples(event_trigger_examples, event_trigger_probs)


def get_role_probs(l_train: pd.DataFrame, lfs: Optional[List[labeling_function]] = None,
                   label_model: LabelModel = None):
    """

    :param l_train:
    :param lfs:
    :param label_model:
    :return:
    """
    event_role_examples, _ = build_event_role_examples(l_train)
    if lfs is None:
        lfs = [
            lf_delay_type,
            lf_direction_type,
            lf_start_location_type,
            lf_end_location_type,
            lf_start_date_type,
            lf_end_date_type,
            lf_cause_type,
            lf_cause_gaz_file,
            lf_distance_type,
            lf_route_type,
            lf_not_an_event,
            lf_somajo_separate_sentence,
            lf_event_patterns,
            lf_event_patterns_general_location
        ]
    applier = PandasLFApplier(lfs)
    df_train = applier.apply(event_role_examples)
    if label_model is None:
        label_model = LabelModel(cardinality=11, verbose=True)
        label_model.fit(L_train=df_train, n_epochs=500, log_freq=100, seed=123)
    event_role_probs = label_model.predict_proba(df_train)
    return merge_event_trigger_examples(event_role_examples, event_role_probs)


def build_training_data(lf_train: pd.DataFrame) -> pd.DataFrame:
    """
    Merges event_trigger_examples and event_role examples to build training data.
    :param lf_train: DataFrame with original data.
    :return: Original DataFrame updated with event triggers and event roles.
    """
    merged_event_trigger_examples = get_trigger_probs(lf_train)
    merged_event_role_examples = get_role_probs(lf_train)

    merged_examples: pd.DataFrame = utils.get_deep_copy(lf_train)
    if 'id' in merged_examples:
        merged_examples.set_index('id', inplace=True)

    merged_examples.update(merged_event_trigger_examples)
    merged_examples.update(merged_event_role_examples)

    return merged_examples
