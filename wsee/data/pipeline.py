import os
import logging
from pathlib import Path
from typing import Optional, List, Any, Dict

import pandas as pd
import numpy as np
from snorkel.labeling import LabelModel, PandasLFApplier, labeling_function, filter_unlabeled_dataframe
from tqdm import tqdm

from wsee.preprocessors import preprocessors
from wsee.labeling import event_trigger_lfs
from wsee.labeling import event_argument_role_lfs
from wsee.utils import utils


logging.basicConfig(level=logging.INFO)


event_type_lf_map: Dict[int, Any] = {
    event_trigger_lfs.Accident: event_trigger_lfs.lf_accident_context,
    event_trigger_lfs.CanceledRoute: event_trigger_lfs.lf_canceledroute_cat,
    event_trigger_lfs.CanceledStop: event_trigger_lfs.lf_canceledstop_cat,
    event_trigger_lfs.Delay: event_trigger_lfs.lf_delay_cat,
    event_trigger_lfs.Obstruction: event_trigger_lfs.lf_obstruction_cat,
    event_trigger_lfs.RailReplacementService: event_trigger_lfs.lf_railreplacementservice_cat,
    event_trigger_lfs.TrafficJam: event_trigger_lfs.lf_trafficjam_cat
}

event_type_location_type_map: Dict[int, List[str]] = {
    event_trigger_lfs.Accident: ['location', 'location_street', 'location_city', 'location_route'],
    event_trigger_lfs.CanceledRoute: ['location_route'],
    event_trigger_lfs.CanceledStop: ['location_stop'],
    event_trigger_lfs.Delay: ['location_route'],
    event_trigger_lfs.Obstruction: ['location', 'location_street', 'location_city', 'location_route'],
    event_trigger_lfs.RailReplacementService: ['location_route'],
    event_trigger_lfs.TrafficJam: ['location', 'location_street', 'location_city', 'location_route']
}


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
        logging.info(f"Reading {split} data from: {sd_path}")
        sd_data = pd.read_json(sd_path, lines=True, encoding='utf8')
        output_dict[split] = sd_data

    daystream_path = os.path.join(input_path, 'daystream.jsonl')
    assert os.path.exists(daystream_path)
    logging.info(f"Reading daystream data from: {daystream_path}")
    daystream = pd.read_json(daystream_path, lines=True, encoding='utf8')
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
    example_count = 0

    logging.info("Building event trigger examples")
    logging.info(f"DataFrame has {len(dataframe.index)} rows")
    for index, row in tqdm(dataframe.iterrows()):
        entity_type_freqs = preprocessors.get_entity_type_freqs(row)
        for event_trigger in row.event_triggers:
            trigger_row = row.copy()
            trigger_row['trigger'] = preprocessors.get_entity(event_trigger['id'], row.entities)
            trigger_row['entity_type_freqs'] = entity_type_freqs
            event_trigger_rows.append(trigger_row)
            event_type_num = np.asarray(event_trigger['event_type_probs']).argmax()
            event_trigger_rows_y.append(event_type_num)
            example_count += 1
            if event_type_num != 7:
                event_count += 1

    if event_count > 0:
        logging.info(f"Number of events: {event_count}")
    logging.info(f"Number of event trigger examples: {example_count}")
    event_trigger_rows = pd.DataFrame(event_trigger_rows)
    event_trigger_rows_y = np.asarray(event_trigger_rows_y)
    return event_trigger_rows, event_trigger_rows_y


def arg_location_type_event_type_match(cand):
    arg_entity_type = cand.argument['entity_type']
    for event_class, location_types in event_type_location_type_map.items():
        if arg_entity_type in location_types and event_type_lf_map[event_class](cand) == event_class:
            return True
    return False


def build_event_role_examples(dataframe):
    """
    Takes a dataframe containing one document per row with all its annotations
    (event roles are of interest here) and creates one row for each trigger-entity
    (event role) pair. Also adds attributes beforehand instead of using preprocessors in
    order not to do it for each row or even each row*labeling functions.
    :param dataframe: Annotated documents.
    :return: DataFrame containing event role examples and NumPy array containing labels.
    """
    event_role_rows_list = []
    event_role_rows_y = []

    event_count = 0
    example_count = 0

    logging.info("Building event role examples")
    logging.info(f"DataFrame has {len(dataframe.index)} rows")
    logging.info("Adding the following attributes to each row: "
                 "entity_type_freqs, somajo_doc, mixed_ner, mixed_ner_spans, not_an_event, arg_type_event_type_match, "
                 "between_distance, is_multiple_same_event_type")
    for index, row in tqdm(dataframe.iterrows()):
        entity_type_freqs = preprocessors.get_entity_type_freqs(row)
        somajo_doc = preprocessors.get_somajo_doc(row)
        mixed_ner, mixed_ner_spans = preprocessors.get_mixed_ner(row)
        for event_role in row.event_roles:
            role_row = row.copy()
            role_row['trigger'] = preprocessors.get_entity(event_role['trigger'], row.entities)
            role_row['argument'] = preprocessors.get_entity(event_role['argument'], row.entities)
            role_row['entity_type_freqs'] = entity_type_freqs
            role_row['somajo_doc'] = somajo_doc
            role_row['separate_sentence'] = preprocessors.get_somajo_separate_sentence(role_row)
            role_row['mixed_ner'] = mixed_ner
            role_row['mixed_ner_spans'] = mixed_ner_spans
            role_row['not_an_event'] = event_trigger_lfs.lf_negative(role_row) == event_trigger_lfs.O
            role_row['arg_location_type_event_type_match'] = arg_location_type_event_type_match(role_row)
            role_row['between_distance'] = preprocessors.get_between_distance(role_row)
            role_row['is_multiple_same_event_type'] = preprocessors.is_multiple_same_event_type(role_row)
            event_role_rows_list.append(role_row)
            event_role_num = np.asarray(event_role['event_argument_probs']).argmax()
            event_role_rows_y.append(event_role_num)
            example_count += 1
            if event_role_num != 10:
                event_count += 1

    if event_count > 0:
        logging.info(f"Number of event roles: {event_count}")
    logging.info(f"Number of event role examples: {example_count}")
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
        'id': x.trigger['id'],
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
    logging.info("Merging event trigger examples that belong to the same document")
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
        'trigger': x.trigger['id'],
        'argument': x.argument['id'],
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
    logging.info("Merging event role examples that belong to the same document")
    event_role_rows_copy = event_role_rows.copy()
    event_role_rows_copy['event_argument_probs'] = list(event_argument_probs)
    event_role_rows_copy = event_role_rows_copy.apply(build_labeled_event_role, axis=1)
    aggregation_functions = {
        'event_roles': 'sum'  # expects list of one event role per row
    }
    return event_role_rows_copy.groupby('id').agg(aggregation_functions)


def get_trigger_probs(l_train: pd.DataFrame, filter_abstains: bool = True,
                      lfs: Optional[List[labeling_function]] = None,
                      label_model: LabelModel = None):
    """
    Takes "raw" data frame, builds trigger examples, (trains LabelModel), calculates event_trigger_probs
    and returns merged trigger examples with event_trigger_probs.
    :param filter_abstains: Filters rows where all labeling functions abstained
    :param l_train:
    :param lfs:
    :param label_model:
    :return:
    """
    df_train, _ = build_event_trigger_examples(l_train)
    if lfs is None:
        lfs = [
            event_trigger_lfs.lf_accident_context,
            event_trigger_lfs.lf_accident_context_street,
            event_trigger_lfs.lf_canceledroute_cat,
            event_trigger_lfs.lf_canceledstop_cat,
            event_trigger_lfs.lf_delay_cat,
            event_trigger_lfs.lf_delay_priorities,
            event_trigger_lfs.lf_delay_duration,
            event_trigger_lfs.lf_obstruction_cat,
            event_trigger_lfs.lf_obstruction_street,
            event_trigger_lfs.lf_obstruction_priorities,
            event_trigger_lfs.lf_railreplacementservice_cat,
            event_trigger_lfs.lf_trafficjam_cat,
            event_trigger_lfs.lf_trafficjam_street,
            event_trigger_lfs.lf_trafficjam_order,
            event_trigger_lfs.lf_negative,
            event_trigger_lfs.lf_cause_negative
        ]
    logging.info("Running Event Trigger Labeling Function Applier")
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)
    logging.info("Fitting LabelModel on the data and predicting class probabilities")
    if label_model is None:
        label_model = LabelModel(cardinality=8, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123,
                        class_balance=[
                            0.07221542227662178,
                            0.07466340269277846,
                            0.030599755201958383,
                            0.0795593635250918,
                            0.12362301101591187,
                            0.02692778457772338,
                            0.189718482252142,
                            0.40269277845777235]
                        )
    event_trigger_probs = label_model.predict_proba(L_train)

    if filter_abstains:
        df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
            X=df_train, y=event_trigger_probs, L=L_train
        )

        return merge_event_trigger_examples(df_train_filtered, probs_train_filtered)
    else:
        return merge_event_trigger_examples(df_train, event_trigger_probs)


def get_role_probs(l_train: pd.DataFrame, filter_abstains: bool = True,
                   lfs: Optional[List[labeling_function]] = None,
                   label_model: LabelModel = None):
    """

    :param filter_abstains: Filters rows where all labeling functions abstained
    :param l_train:
    :param lfs:
    :param label_model:
    :return:
    """
    df_train, _ = build_event_role_examples(l_train)
    if lfs is None:
        lfs = [
            event_argument_role_lfs.lf_location_same_sentence_is_event,
            event_argument_role_lfs.lf_location_same_sentence_nearest_is_event,
            event_argument_role_lfs.lf_location_chained,
            event_argument_role_lfs.lf_location_adjacent_markers,
            event_argument_role_lfs.lf_location_beginning_street_stop_route,
            event_argument_role_lfs.lf_location_first_sentence,
            event_argument_role_lfs.lf_location_first_sentence_nearest,
            event_argument_role_lfs.lf_location_first_sentence_street_stop_route,
            event_argument_role_lfs.lf_location_first_sentence_priorities,
            event_argument_role_lfs.lf_delay_event_sentence,
            event_argument_role_lfs.lf_delay_event_sentence_check,
            event_argument_role_lfs.lf_direction_type,
            event_argument_role_lfs.lf_direction_order,
            event_argument_role_lfs.lf_start_location_type,
            event_argument_role_lfs.lf_start_location_nearest,
            event_argument_role_lfs.lf_end_location_type,
            event_argument_role_lfs.lf_end_location_nearest,
            event_argument_role_lfs.lf_start_date_type,
            event_argument_role_lfs.lf_start_date_first,
            event_argument_role_lfs.lf_start_date_adjacent,
            event_argument_role_lfs.lf_end_date_type,
            event_argument_role_lfs.lf_cause_type,
            event_argument_role_lfs.lf_cause_order,
            event_argument_role_lfs.lf_cause_gaz_file,
            event_argument_role_lfs.lf_distance_type,
            event_argument_role_lfs.lf_distance_nearest,
            event_argument_role_lfs.lf_route_type,
            event_argument_role_lfs.lf_route_type_order,
            event_argument_role_lfs.lf_not_an_event,
            event_argument_role_lfs.lf_somajo_separate_sentence,
            event_argument_role_lfs.lf_overlapping,
            event_argument_role_lfs.lf_too_far_40,
            event_argument_role_lfs.lf_multiple_same_event_type,
            event_argument_role_lfs.lf_event_patterns,
            event_argument_role_lfs.lf_event_patterns_general_location
        ]
    logging.info("Running Event Role Labeling Function Applier")
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)
    logging.info("Fitting LabelModel on the data and predicting class probabilities")
    if label_model is None:
        label_model = LabelModel(cardinality=11, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123,
                        class_balance=[
                            0.07511483382869495,
                            0.010537692515536342,
                            0.037017022426371254,
                            0.04998649013780059,
                            0.0466090245879492,
                            0.0045933531477978925,
                            0.0054039448797622265,
                            0.013915158065387734,
                            0.018238313969197513,
                            0.0031072683058632803,
                            0.735476898135639
                        ])
    event_role_probs = label_model.predict_proba(L_train)

    if filter_abstains:
        df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
            X=df_train, y=event_role_probs, L=L_train
        )

        return merge_event_role_examples(df_train_filtered, probs_train_filtered)
    else:
        return merge_event_role_examples(df_train, event_role_probs)


def build_training_data(lf_train: pd.DataFrame, save_path=None, sample=False) -> pd.DataFrame:
    """
    Merges event_trigger_examples and event_role examples to build training data.
    :param sample: When set to true, only use a sample of the data.
    :param save_path: Where to save the dataframe as a jsonl
    :param lf_train: DataFrame with original data.
    :return: Original DataFrame updated with event triggers and event roles.
    """
    if sample and len(lf_train) > 100:
        lf_train = lf_train.sample(100)
    merged_event_trigger_examples = get_trigger_probs(lf_train)

    if save_path:
        try:
            logging.info(f"Writing Snorkel Trigger data to {save_path + '/daystream_triggers.jsonl'}")
            merged_event_trigger_examples.reset_index(level=0).to_json(
                save_path + '/daystream_triggers.jsonl', orient='records', lines=True, force_ascii=False)
        except Exception as e:
            print(e)

    merged_event_role_examples = get_role_probs(lf_train)

    if save_path:
        try:
            logging.info(f"Writing Snorkel Role data to {save_path + '/daystream_roles.jsonl'}")
            merged_event_role_examples.reset_index(level=0).to_json(
                save_path + '/daystream_roles.jsonl', orient='records', lines=True, force_ascii=False)
        except Exception as e:
            print(e)

    merged_examples: pd.DataFrame = utils.get_deep_copy(lf_train)
    # Make sure to remove event_triggers and roles that were built per default
    for idx, row in merged_examples.iterrows():
        row['event_triggers'] = []
        row['event_roles'] = []
    if 'id' in merged_examples:
        merged_examples.set_index('id', inplace=True)

    merged_examples.update(merged_event_trigger_examples)
    merged_examples.update(merged_event_role_examples)

    merged_examples.reset_index(level=0, inplace=True)

    # Removes rows with no events
    merged_examples = merged_examples[merged_examples['event_triggers'].map(lambda d: len(d)) > 0]

    if save_path:
        try:
            logging.info(f"Writing Snorkel Labeled data to {save_path+'/daystream_snorkeledv6_pipeline.jsonl'}")
            merged_examples.to_json(
                save_path + '/daystream_snorkeledv6_pipeline.jsonl', orient='records', lines=True, force_ascii=False)
        except Exception as e:
            print(e)
    return merged_examples
