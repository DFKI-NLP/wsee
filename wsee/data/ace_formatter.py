import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from wsee import SD4M_RELATION_TYPES, ROLE_LABELS, NEGATIVE_TRIGGER_LABEL, NEGATIVE_ARGUMENT_LABEL
from wsee.preprocessors import preprocessors


def main(args):
    input_path = Path(args.input)
    assert input_path.exists(), 'Input not found: %s'.format(args.input)
    snorkel_file = pd.read_json(input_path, lines=True, encoding='utf8')
    ace_file: pd.DataFrame = snorkel_to_ace_format(snorkel_file)
    ace_file.drop(labels=['event_triggers', 'event_roles'], axis=1, inplace=True)
    ace_file.to_json(Path(args.output), orient='records', lines=True, force_ascii=False)


def snorkel_to_ace_format(df: pd.DataFrame) -> pd.DataFrame:
    assert 'event_triggers' in df
    assert 'event_roles' in df
    df_copy = df.copy()
    return df_copy.apply(create_events, axis=1)


def create_events(row: pd.Series) -> pd.Series:
    formatted_events = []
    if 'entities' in row and 'event_triggers' in row and 'event_roles' in row:
        filtered_triggers_with_labels = [
            (trigger, SD4M_RELATION_TYPES[np.asarray(trigger['event_type_probs']).argmax()])
            for trigger in row['event_triggers']
            if np.asarray(trigger['event_type_probs']).sum() > 0.0 and
            SD4M_RELATION_TYPES[np.asarray(trigger['event_type_probs']).argmax()] != NEGATIVE_TRIGGER_LABEL]
        filtered_roles_with_labels = [
            (role, ROLE_LABELS[np.asarray(role['event_argument_probs']).argmax()])
            for role in row['event_roles']
            if np.asarray(role['event_argument_probs']).sum() > 0.0 and
            ROLE_LABELS[np.asarray(role['event_argument_probs']).argmax()] != NEGATIVE_ARGUMENT_LABEL]

        for event_trigger, trigger_label in filtered_triggers_with_labels:
            trigger_entity = preprocessors.get_entity(event_trigger['id'], row['entities'])
            event_type = trigger_label
            formatted_trigger = {
                'id': trigger_entity['id'],
                'text': trigger_entity['text'],
                'entity_type': trigger_entity['entity_type'],
                'start': trigger_entity['start'],
                'end': trigger_entity['end']
            }
            relevant_args_with_labels = [(arg, label) for arg, label in filtered_roles_with_labels if
                                         arg['trigger'] == event_trigger['id']]
            formatted_args = []
            for event_arg, role_label in relevant_args_with_labels:
                event_arg_entity = preprocessors.get_entity(event_arg['argument'], row['entities'])
                arg_role = role_label
                formatted_arg = {
                    'id': event_arg_entity['id'],
                    'text': event_arg_entity['text'],
                    'entity_type': event_arg_entity['entity_type'],
                    'start': event_arg_entity['start'],
                    'end': event_arg_entity['end'],
                    'role': arg_role
                }
                formatted_args.append(formatted_arg)
            formatted_event = {
                'event_type': event_type,
                'trigger': formatted_trigger,
                'arguments': formatted_args
            }
            formatted_events.append(formatted_event)
    row['events'] = formatted_events
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Snorkel jsonl format to ACE jsonl format')
    parser.add_argument('--input', type=str, help='Input path')
    parser.add_argument('--output', type=str, help='Output path')
    arguments = parser.parse_args()
    main(arguments)
