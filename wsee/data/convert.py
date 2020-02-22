import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from fastavro import reader
from tqdm import tqdm
from typing import Dict

NEGATIVE_TRIGGER_LABEL = 'O'
NEGATIVE_ARGUMENT_LABEL = 'no_arg'

SD4M_RELATION_TYPES = ['Accident', 'CanceledRoute', 'CanceledStop', 'Delay',
                       'Obstruction', 'RailReplacementService', 'TrafficJam',
                       NEGATIVE_TRIGGER_LABEL]
# if we use their indices (-1), we might want to move 'Other' to the beginning
ROLE_LABELS = ['location', 'delay', 'direction',
               'start_loc', 'end_loc',
               'start_date', 'end_date', 'cause',
               'jam_length', 'route', NEGATIVE_ARGUMENT_LABEL]


def main(args):
    input_path = Path(args.input)
    assert input_path.exists(), 'Input not found: %s'.format(args.input)

    one_hot = False if args.no_one_hot else True

    daystream = []
    for split in ['train', 'dev', 'test']:
        input_file = input_path.joinpath(split, f'{split}.avro')
        output_file = input_path.joinpath(split, f'{split}_with_events.jsonl')
        smart_data, daystream_part = convert_file(input_file, one_hot)
        smart_data.to_json(output_file, orient='records', lines=True, force_ascii=False)
        daystream.append(daystream_part)
    daystream = pd.concat(daystream, sort=False).reset_index(drop=True)
    daystream.to_json(input_path.joinpath('daystream.jsonl'), orient='records', lines=True, force_ascii=False)


def convert_file(file_path, one_hot=False):
    sd_list = []
    daystream_list = []
    with open(file_path, 'rb') as input_file:
        avro_reader = reader(input_file)
        for doc in tqdm(avro_reader):
            is_smart_data_doc = 'fileName' in doc['refids'] and doc['refids']['fileName'].startswith('smartdata')
            if is_smart_data_doc:
                converted_doc = convert_doc(doc=doc, one_hot=one_hot)
                if converted_doc:
                    sd_list.append(converted_doc)
            else:
                converted_doc = convert_doc(doc=doc, one_hot=one_hot)
                if converted_doc:
                    daystream_list.append(converted_doc)
    smart_data = pd.DataFrame(sd_list)
    daystream = pd.DataFrame(daystream_list)
    return smart_data, daystream


def convert_entity(text, tokens, entity):
    try:
        start_token_idx, end_token_idx = span_to_token_idxs(tokens, entity['span'])
        return {
            'id': entity['id'],
            'text': span_to_text(text, entity['span']),
            'entity_type': entity['type'],
            'start': start_token_idx,
            'end': end_token_idx + 1  # Generate exclusive token spans
        }
    except StopIteration:
        print('Token offset issue')
        print(tokens, '\n', entity)
        print('Does this information help?')
        return None


def span_to_text(text, span):
    return text[span['start']:span['end']]


def span_to_token_idxs(tokens, span):
    start_idx = next(idx
                     for idx, token in enumerate(tokens)
                     if token['span']['start'] == span['start'])
    end_idx = next(idx
                   for idx, token in enumerate(tokens)
                   if token['span']['end'] == span['end'])
    return start_idx, end_idx


def get_index_for_id(obj_id, obj_list):
    for idx, obj in enumerate(obj_list):
        if obj['id'] == obj_id:
            return idx
    raise Exception(f'obj_id was not found in obj_list. The value of obj_id was: {obj_id}.\nThe obj_list: {obj_list}.')


def convert_doc(doc: Dict, doc_text: str = None, one_hot=False):
    if doc_text is None:
        try:
            doc_text = str(doc['text'])
        except KeyError:
            print(doc['id'], "gave KeyError for doc['text'] and doc_text was None.")
            return None
    s_id = doc['id']
    text = span_to_text(doc_text, doc['span']) if 'span' in doc else doc_text
    tokens = [span_to_text(doc_text, token['span']) for token in
              doc['tokens']]
    # pos_tags = [token['posTag'] for token in doc['tokens']]
    ner_tags = [token['ner'] for token in doc['tokens']]

    entities = []
    for cm in doc['conceptMentions']:
        converted_entity = convert_entity(text=doc_text, tokens=doc['tokens'], entity=cm)
        if converted_entity:
            entities.append(converted_entity)

    event_triggers = []
    event_roles = []

    # TODO check how to handle this for predict task
    # I initially set the fillers to None in case the document did not have relationMentions
    if one_hot:
        trigger_filler = one_hot_encode(NEGATIVE_TRIGGER_LABEL, SD4M_RELATION_TYPES)
        arg_role_filler = one_hot_encode(NEGATIVE_ARGUMENT_LABEL, ROLE_LABELS)
    else:
        trigger_filler = NEGATIVE_TRIGGER_LABEL
        arg_role_filler = NEGATIVE_ARGUMENT_LABEL

    # Set up all possible triggers with default trigger label
    for entity in entities:
        if entity['entity_type'] in ['TRIGGER', 'trigger']:
            if one_hot:
                event_triggers.append({'id': entity['id'],
                                       'event_type_probs': trigger_filler})
            else:
                event_triggers.append({'id': entity['id'], 'event_type': trigger_filler})

    # Build all possible pairs of triggers and entities with default argument role label
    for trigger in event_triggers:
        for entity in entities:
            if trigger['id'] != entity['id']:
                if one_hot:
                    event_roles.append({
                        'trigger': trigger['id'],
                        'argument': entity['id'],
                        'event_argument_probs': arg_role_filler})
                else:
                    event_roles.append({
                        'trigger': trigger['id'],
                        'argument': entity['id'],
                        'event_argument': arg_role_filler})

    # Update event types and event argument roles
    if doc['relationMentions']:
        filtered_relations = [
            rm for rm in doc['relationMentions'] if rm['name'] in SD4M_RELATION_TYPES
        ]
        for rm in filtered_relations:
            # collect trigger(s) in relation mention
            trigger = next((arg for arg in rm['args'] if arg['role'] == 'trigger'), None)
            if trigger is None or trigger['conceptMention']['type'] not in ['TRIGGER', 'trigger']:
                print('Skipping invalid event')
                continue
            trigger_id = trigger['conceptMention']['id']
            # update event type (probs) in event_triggers
            try:
                trigger_idx = get_index_for_id(trigger_id, event_triggers)
                if one_hot:
                    event_triggers[trigger_idx]['event_type_probs'] = one_hot_encode(
                        rm['name'],
                        SD4M_RELATION_TYPES,
                        NEGATIVE_TRIGGER_LABEL
                    )
                else:
                    event_triggers[trigger_idx]['event_type'] = rm['name']
            except Exception as e:
                print(f'{e}\n See entity list: {entities}.')
                continue

            role_args = [arg for arg in rm['args'] if arg['role'] != 'trigger']
            for event_role in event_roles:
                if event_role['trigger'] == trigger_id:
                    arg_role = next((arg['role'] for arg in role_args
                                     if arg['conceptMention']['id'] == event_role['argument']),
                                    NEGATIVE_ARGUMENT_LABEL)
                    if one_hot:
                        event_role['event_argument_probs'] = one_hot_encode(
                            arg_role,
                            ROLE_LABELS,
                            NEGATIVE_ARGUMENT_LABEL
                        )
                    else:
                        event_role['event_argument'] = arg_role

    return {'id': s_id, 'text': text, 'tokens': tokens,
            # 'pos-tag': pos_tag,
            'ner_tags': ner_tags, 'entities': entities,
            'event_triggers': event_triggers, 'event_roles': event_roles}


def convert_ner_tag(ner):
    return ner.replace('-', '_').upper()


def one_hot_encode(label, label_names, negative_label='O'):
    label = label if label in label_names else negative_label
    class_probs = [1.0 if label_name == label else 0.0 for label_name in label_names]
    return class_probs


def one_hot_decode(class_probs, label_names):
    class_probs_array = np.asarray(class_probs)
    class_name = label_names[class_probs_array.argmax()]
    return class_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smartdata avro schema to jsonl converter')
    parser.add_argument('--input', type=str, help='Directory of avro file structure')
    parser.add_argument("--no_one_hot", action="store_true", help='Do not one hot encode labels')
    parser.add_argument('-f', dest='force_output', action='store_true',
                        help='Force creation of new output folder')
    arguments = parser.parse_args()
    main(arguments)
