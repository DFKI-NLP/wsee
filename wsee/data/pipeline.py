import argparse
import os
import logging
from pathlib import Path
from typing import Optional, List, Any, Dict

import pandas as pd
import numpy as np
from snorkel.labeling import LabelModel, PandasLFApplier, labeling_function, filter_unlabeled_dataframe
from tqdm import tqdm
from multiprocessing import Pool

from wsee.preprocessors import preprocessors
from wsee.labeling import event_trigger_lfs as trigger_lfs
from wsee.labeling import event_argument_role_lfs as role_lfs
from wsee.utils import utils
from wsee import SD4M_RELATION_TYPES, ROLE_LABELS, NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL

logging.basicConfig(level=logging.INFO)

event_type_lf_map: Dict[int, Any] = {
    trigger_lfs.Accident: trigger_lfs.lf_accident_chained,
    trigger_lfs.CanceledRoute: trigger_lfs.lf_canceledroute_cat,
    trigger_lfs.CanceledStop: trigger_lfs.lf_canceledstop_cat,
    trigger_lfs.Delay: trigger_lfs.lf_delay_chained,
    trigger_lfs.Obstruction: trigger_lfs.lf_obstruction_chained,
    trigger_lfs.RailReplacementService: trigger_lfs.lf_railreplacementservice_cat,
    trigger_lfs.TrafficJam: trigger_lfs.lf_trafficjam_chained
}

event_type_location_type_map: Dict[int, List[str]] = {
    trigger_lfs.Accident: ['location', 'location_street', 'location_city', 'location_route'],
    trigger_lfs.CanceledRoute: ['location_route'],
    trigger_lfs.CanceledStop: ['location_stop'],
    trigger_lfs.Delay: ['location_route'],
    trigger_lfs.Obstruction: ['location', 'location_street', 'location_city', 'location_route'],
    trigger_lfs.RailReplacementService: ['location_route'],
    trigger_lfs.TrafficJam: ['location', 'location_street', 'location_city', 'location_route']
}


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def preprocess_docs_for_roles(doc):
    entity_type_freqs = preprocessors.get_entity_type_freqs(doc)
    somajo_doc = preprocessors.get_somajo_doc(doc)
    mixed_ner, mixed_ner_spans = preprocessors.get_mixed_ner(doc)
    doc['entity_type_freqs'] = entity_type_freqs
    doc['somajo_doc'] = somajo_doc
    doc['mixed_ner'] = mixed_ner
    doc['mixed_ner_spans'] = mixed_ner_spans
    return doc


def preprocess_docs_for_roles_applier(df):
    return df.apply(lambda doc: preprocess_docs_for_roles(doc), axis=1)


def preprocess_role_examples(role_row):
    role_row['separate_sentence'] = preprocessors.get_somajo_separate_sentence(role_row)
    role_row['not_an_event'] = trigger_lfs.lf_negative(role_row) == trigger_lfs.O
    role_row['arg_location_type_event_type_match'] = arg_location_type_event_type_match(role_row)
    role_row['between_distance'] = preprocessors.get_between_distance(role_row)
    role_row['is_multiple_same_event_type'] = preprocessors.is_multiple_same_event_type(role_row)
    return role_row


def preprocess_role_examples_applier(df):
    return df.apply(lambda doc: preprocess_role_examples(doc), axis=1)


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

    event_trigger_rows = pd.DataFrame(event_trigger_rows)
    event_trigger_rows_y = np.asarray(event_trigger_rows_y)

    label, count = np.unique(event_trigger_rows_y, return_counts=True)
    label_count_map = dict(zip(label, count))
    negative_trigger_idx = SD4M_RELATION_TYPES.index(NEGATIVE_TRIGGER_LABEL)
    if negative_trigger_idx in label_count_map:
        logging.info(f"Number of events: {len(event_trigger_rows)-label_count_map[negative_trigger_idx]}")
    logging.info(f"Number of event trigger examples: {len(event_trigger_rows)}")
    return event_trigger_rows, event_trigger_rows_y


def arg_location_type_event_type_match(cand):
    arg_entity_type = cand.argument['entity_type']
    for event_class, location_types in event_type_location_type_map.items():
        if arg_entity_type in location_types and event_type_lf_map[event_class](cand) == event_class:
            return True
    return False


def build_event_role_examples(dataframe, n_cores=4):
    """
    Takes a dataframe containing one document per row with all its annotations
    (event roles are of interest here) and creates one row for each trigger-entity
    (event role) pair. Also adds attributes beforehand instead of using preprocessors in
    order not to do it for each row or even each row*labeling functions.
    :param n_cores: Number of cores to process dataframe in parallel.
    :param dataframe: Annotated documents.
    :return: DataFrame containing event role examples and NumPy array containing labels.
    """
    event_role_rows_list = []
    event_role_rows_y = []

    logging.info("Building event role examples")
    logging.info(f"DataFrame has {len(dataframe.index)} rows")
    logging.info("Adding the following attributes to each document: "
                 "entity_type_freqs, somajo_doc, mixed_ner, mixed_ner_spans")

    # 1. Preprocess docs (entity frequencies, sentence splitting, mixed ner pattern)
    dataframe = parallelize_dataframe(dataframe, preprocess_docs_for_roles_applier, n_cores=n_cores)

    # 2. Build role examples
    for index, row in tqdm(dataframe.iterrows()):
        for event_role in row.event_roles:
            role_row = row.copy()
            role_row['trigger'] = preprocessors.get_entity(event_role['trigger'], row.entities)
            role_row['argument'] = preprocessors.get_entity(event_role['argument'], row.entities)
            event_role_rows_list.append(role_row)
            event_role_num = np.asarray(event_role['event_argument_probs']).argmax()
            event_role_rows_y.append(event_role_num)
    event_role_rows = pd.DataFrame(event_role_rows_list).reset_index(drop=True)
    event_role_rows_y = np.asarray(event_role_rows_y)

    # 3. Process role examples (not_an_event, arg_type_event_type_match, between_distance, is_multiple_same_event_type)
    logging.info("Adding the following attributes to each role example: "
                 "not_an_event, arg_type_event_type_match, between_distance, is_multiple_same_event_type")
    event_role_rows = parallelize_dataframe(event_role_rows, preprocess_role_examples_applier, n_cores=n_cores)

    label, count = np.unique(event_role_rows_y, return_counts=True)
    label_count_map = dict(zip(label, count))
    negative_trigger_idx = ROLE_LABELS.index(NEGATIVE_ARGUMENT_LABEL)
    if negative_trigger_idx in label_count_map:
        logging.info(f"Number of event roles: {len(event_role_rows) - label_count_map[negative_trigger_idx]}")
    logging.info(f"Number of event role examples: {len(event_role_rows)}")
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
    if 'event_triggers' in event_trigger_rows:
        event_trigger_rows.drop('event_triggers', axis=1, inplace=True)
    event_trigger_rows = event_trigger_rows.apply(build_labeled_event_trigger, axis=1)
    aggregation_functions = {
        'text': 'first',
        'tokens': 'first',
        # 'pos_tags': 'first',
        'ner_tags': 'first',
        'entities': 'first',
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
    if 'event_roles' in event_role_rows:
        event_role_rows.drop('event_roles', axis=1, inplace=True)
    event_role_rows['event_argument_probs'] = list(event_argument_probs)
    event_role_rows = event_role_rows.apply(build_labeled_event_role, axis=1)
    aggregation_functions = {
        'text': 'first',
        'tokens': 'first',
        # 'pos_tags': 'first',
        'ner_tags': 'first',
        'entities': 'first',
        'event_roles': 'sum'  # expects list of one event role per row
    }
    return event_role_rows.groupby('id').agg(aggregation_functions)


def get_trigger_probs(lf_train: pd.DataFrame, filter_abstains: bool = False,
                      lfs: Optional[List[labeling_function]] = None,
                      lf_dev: pd.DataFrame = None):
    """
    Takes "raw" data frame, builds trigger examples, (trains LabelModel), calculates event_trigger_probs
    and returns merged trigger examples with event_trigger_probs.
    :param filter_abstains: Filters rows where all labeling functions abstained
    :param lf_train: Training dataset which will be labeled using Snorkel
    :param lfs: List of labeling functions
    :param lf_dev: Optional development dataset that can be used to set a prior for the class balance
    :return:
    """
    df_train, _ = build_event_trigger_examples(lf_train)
    df_dev, Y_dev = None, None
    if lf_dev is not None:
        df_dev, Y_dev = build_event_trigger_examples(lf_dev)
    if lfs is None:
        lfs = [
            trigger_lfs.lf_accident_context,
            trigger_lfs.lf_accident_context_street,
            trigger_lfs.lf_accident_context_no_cause_check,
            trigger_lfs.lf_canceledroute_cat,
            trigger_lfs.lf_canceledroute_replicated,
            trigger_lfs.lf_canceledstop_cat,
            trigger_lfs.lf_canceledstop_replicated,
            trigger_lfs.lf_delay_cat,
            trigger_lfs.lf_delay_priorities,
            trigger_lfs.lf_delay_duration,
            trigger_lfs.lf_obstruction_cat,
            trigger_lfs.lf_obstruction_street,
            trigger_lfs.lf_obstruction_priorities,
            trigger_lfs.lf_railreplacementservice_cat,
            trigger_lfs.lf_railreplacementservice_replicated,
            trigger_lfs.lf_trafficjam_cat,
            trigger_lfs.lf_trafficjam_street,
            trigger_lfs.lf_trafficjam_order,
            trigger_lfs.lf_negative,
            trigger_lfs.lf_cause_negative,
            trigger_lfs.lf_obstruction_negative
        ]
    logging.info("Running Event Trigger Labeling Function Applier")
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)
    logging.info("Fitting LabelModel on the data and predicting trigger class probabilities")
    label_model = LabelModel(cardinality=8, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=5000, log_freq=500, seed=12345, Y_dev=Y_dev)

    # Evaluate label model on development data
    if df_dev is not None and Y_dev is not None:
        logging.info("Running Event Role Labeling Function Applier on dev set and evaluate label model")
        L_dev = applier.apply(df_dev)
        label_model_accuracy = label_model.score(L=L_dev, Y=Y_dev, tie_break_policy="random")[
            "accuracy"
        ]
        logging.info(f"{'Trigger Label Model Accuracy:':<25} {label_model_accuracy * 100:.1f}%")

    event_trigger_probs = label_model.predict_proba(L_train)

    if filter_abstains:
        df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
            X=df_train, y=event_trigger_probs, L=L_train
        )

        return merge_event_trigger_examples(df_train_filtered, probs_train_filtered)
    else:
        # Multiplies probabilities of abstains with zero so that the example is treated as padding in the end model
        return merge_event_trigger_examples(df_train, utils.zero_out_abstains(event_trigger_probs, L_train))


def get_role_probs(lf_train: pd.DataFrame, filter_abstains: bool = False,
                   lfs: Optional[List[labeling_function]] = None,
                   lf_dev: pd.DataFrame = None):
    """

    :param filter_abstains: Filters rows where all labeling functions abstained
    :param lf_train: Training dataset which will be labeled using Snorkel
    :param lfs: List of labeling functions
    :param lf_dev: Optional development dataset that can be used to set a prior for the class balance
    :return:
    """
    df_train, _ = build_event_role_examples(lf_train)
    df_dev, Y_dev = None, None
    if lf_dev is not None:
        df_dev, Y_dev = build_event_role_examples(lf_dev)
    if lfs is None:
        lfs = [
            role_lfs.lf_location_adjacent_markers,
            role_lfs.lf_location_adjacent_trigger_verb,
            role_lfs.lf_location_beginning_street_stop_route,
            role_lfs.lf_location_first_sentence_street_stop_route,
            role_lfs.lf_location_first_sentence_priorities,
            role_lfs.lf_delay_event_sentence,
            role_lfs.lf_delay_preceding_arg,
            role_lfs.lf_delay_preceding_trigger,
            role_lfs.lf_direction_markers,
            role_lfs.lf_direction_markers_order,
            role_lfs.lf_direction_pattern,
            role_lfs.lf_direction_markers_pattern_order,
            role_lfs.lf_start_location_type,
            role_lfs.lf_start_location_nearest,
            role_lfs.lf_start_location_preceding_arg,
            role_lfs.lf_end_location_type,
            role_lfs.lf_end_location_nearest,
            role_lfs.lf_end_location_preceding_arg,
            role_lfs.lf_start_date_type,
            role_lfs.lf_start_date_replicated,
            role_lfs.lf_start_date_first,
            role_lfs.lf_start_date_adjacent,
            role_lfs.lf_end_date_type,
            role_lfs.lf_end_date_replicated,
            role_lfs.lf_cause_type,
            role_lfs.lf_cause_replicated,
            role_lfs.lf_cause_order,
            role_lfs.lf_distance_type,
            role_lfs.lf_distance_nearest,
            role_lfs.lf_distance_order,
            role_lfs.lf_route_type,
            role_lfs.lf_route_type_order,
            role_lfs.lf_route_type_order_between_check,
            role_lfs.lf_delay_earlier_negative,
            role_lfs.lf_date_negative,
            role_lfs.lf_not_an_event,
            role_lfs.lf_somajo_separate_sentence,
            role_lfs.lf_overlapping,
            role_lfs.lf_too_far_40,
            role_lfs.lf_multiple_same_event_type,
            role_lfs.lf_event_patterns,
            role_lfs.lf_event_patterns_general_location
        ]
    logging.info("Running Event Role Labeling Function Applier")
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)
    logging.info("Fitting LabelModel on the data and predicting role class probabilities")
    label_model = LabelModel(cardinality=11, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=5000, log_freq=500, seed=12345, Y_dev=Y_dev)

    # Evaluate label model on development data
    if df_dev is not None and Y_dev is not None:
        logging.info("Running Event Role Labeling Function Applier on dev set and evaluate label model")
        L_dev = applier.apply(df_dev)
        label_model_accuracy = label_model.score(L=L_dev, Y=Y_dev, tie_break_policy="random")[
            "accuracy"
        ]
        logging.info(f"{'Role Label Model Accuracy:':<25} {label_model_accuracy * 100:.1f}%")

    event_role_probs = label_model.predict_proba(L_train)

    if filter_abstains:
        df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
            X=df_train, y=event_role_probs, L=L_train
        )

        return merge_event_role_examples(df_train_filtered, probs_train_filtered)
    else:
        # Multiplies probabilities of abstains with zero so that the example is treated as padding in the end model
        return merge_event_role_examples(df_train, utils.zero_out_abstains(event_role_probs, L_train))


def build_training_data(lf_train: pd.DataFrame, save_path=None,
                        lf_dev: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merges event_trigger_examples and event_role examples to build training data.
    :param save_path: Where to save the dataframe as a jsonl
    :param lf_train: DataFrame with original data.
    :param lf_dev: DataFrame with gold labels, which can be used to estimate the class balance for triggers & roles
    :return: Original DataFrame updated with event triggers and event roles.
    """

    # Trigger labeling
    merged_event_trigger_examples = get_trigger_probs(lf_train=lf_train, lf_dev=lf_dev)

    if save_path:
        try:
            trigger_save_path = Path(save_path).joinpath('daystream_triggers.jsonl')
            logging.info(f"Writing Snorkel Trigger data to {trigger_save_path}")
            merged_event_trigger_examples.reset_index(level=0).to_json(
                trigger_save_path, orient='records', lines=True, force_ascii=False)
        except Exception as e:
            print(e)

    # Role labeling
    merged_event_role_examples = get_role_probs(lf_train=lf_train, lf_dev=lf_dev)

    if save_path:
        try:
            role_save_path = Path(save_path).joinpath('daystream_roles.jsonl')
            logging.info(f"Writing Snorkel Role data to {role_save_path}")
            merged_event_role_examples.reset_index(level=0).to_json(
                role_save_path, orient='records', lines=True, force_ascii=False)
        except Exception as e:
            print(e)

    # Merge
    merged_examples: pd.DataFrame = utils.get_deep_copy(lf_train)
    # Make sure to remove event_triggers and roles that were built per default during the avro-json conversion
    for idx, row in merged_examples.iterrows():
        row['event_triggers'] = []
        row['event_roles'] = []
    if 'id' in merged_examples:
        merged_examples.set_index('id', inplace=True)

    merged_examples.update(
        merged_event_trigger_examples.drop(['text', 'tokens', 'ner_tags', 'entities'], axis=1, inplace=True))
    merged_examples.update(
        merged_event_role_examples.drop(['text', 'tokens', 'ner_tags', 'entities'], axis=1, inplace=True))

    merged_examples.reset_index(level=0, inplace=True)

    # Removes rows with no events/ no positively labeled events
    examples_with_events = list(merged_examples.apply(
        lambda document: utils.has_events(document, include_negatives=True), axis=1))
    logging.info(f"Keeping {sum(examples_with_events)} from {len(merged_examples)} documents with events")
    merged_examples = merged_examples[examples_with_events]

    if save_path:
        try:
            final_save_path = Path(save_path).joinpath("daystream_snorkeled.jsonl")
            logging.info(f"Writing Snorkel Labeled data to {final_save_path}")
            merged_examples.to_json(final_save_path, orient='records', lines=True, force_ascii=False)
        except Exception as e:
            print(e)
    return merged_examples


def main(args):
    input_path = Path(args.input_path)
    assert input_path.exists(), 'Input not found: %s'.format(args.input_path)

    save_path = Path(args.save_path)
    assert save_path.exists(), 'Save path not found: %s'.format(args.save_path)

    loaded_data = load_data(input_path)
    # We label the daystream data with Snorkel and use the train data from SD4M
    labeled_examples = build_training_data(lf_train=loaded_data['daystream'], save_path=save_path,
                                           lf_dev=loaded_data['train'])

    sd_train = loaded_data['train']
    merged = pd.concat([labeled_examples, sd_train])
    merged.to_json(save_path.joinpath('snorkeled_gold_conv_merge_with_abstains.jsonl'),
                   orient='records', lines=True, force_ascii=False)

    logging.info(f"Finished labeling {len(labeled_examples)} documents.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Snorkel event extraction labeler')
    parser.add_argument('--input_path', type=str, help='Path to corpus')
    parser.add_argument('--save_path', type=str, help='Save path for labeled train data')
    arguments = parser.parse_args()
    main(arguments)
