import argparse
from pathlib import Path

import pandas as pd
from fastavro import reader
from tqdm import tqdm
from typing import Dict

from wsee import NEGATIVE_ARGUMENT_LABEL, NEGATIVE_TRIGGER_LABEL, SD4M_RELATION_TYPES, ROLE_LABELS
from wsee.utils import encode


def main(args):
    input_path = Path(args.input)
    assert input_path.exists(), 'Input not found: %s'.format(args.input)

    one_hot = args.one_hot
    build_defaults = args.build_default_events

    daystream = []
    for split in ['train', 'dev', 'test']:
        input_file = input_path.joinpath(split, f'{split}.avro')
        if build_defaults:
            output_file = input_path.joinpath(split, f'{split}_with_events_and_defaults.jsonl')
        else:
            output_file = input_path.joinpath(split, f'{split}_with_events.jsonl')
        smart_data, daystream_part = convert_file(input_file, one_hot, build_defaults)
        smart_data.to_json(output_file, orient='records', lines=True, force_ascii=False)
        daystream.append(daystream_part)
    daystream = pd.concat(daystream, sort=False).reset_index(drop=True)
    daystream_output_file = input_path.joinpath('daystream.jsonl')
    daystream.to_json(daystream_output_file, orient='records', lines=True, force_ascii=False)


def convert_file(file_path, one_hot=False, build_defaults=False):
    sd_list = []
    daystream_list = []
    with open(file_path, 'rb') as input_file:
        avro_reader = reader(input_file)
        for doc in tqdm(avro_reader):
            converted_doc = convert_doc(doc=doc, one_hot=one_hot, build_defaults=build_defaults)
            if converted_doc:
                if _is_smart_data_doc(doc):
                    sd_list.append(converted_doc)
                else:
                    daystream_list.append(converted_doc)
    smart_data = pd.DataFrame(sd_list)
    daystream = pd.DataFrame(daystream_list)
    return smart_data, daystream


def convert_doc(doc: Dict, doc_text: str = None, one_hot=False, build_defaults=False):
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
    ner_tags = _convert_ner_tags([token['ner'] for token in doc['tokens']])
    pos_tags = [token['posTag'] for token in doc['tokens']]

    entities = []
    for cm in doc['conceptMentions']:
        converted_entity = convert_entity(text=doc_text, tokens=doc['tokens'], entity=cm)
        if converted_entity:
            entities.append(converted_entity)

    if _is_smart_data_doc(doc):
        if build_defaults:
            # 1. Does it make sense to build default events and update them, i.e. create negative examples?
            event_triggers, event_roles = build_default_events(entities, one_hot)
            if doc['relationMentions']:
                event_triggers, event_roles = update_events(event_triggers, event_roles, doc['relationMentions'],
                                                            one_hot)
        else:
            # 2. Or does it make more sense to only create events for corresponding relation mentions in doc?
            event_triggers, event_roles = get_events(doc['relationMentions'], one_hot)

    else:
        # Daystream documents do not have relation mention annotation
        event_triggers, event_roles = build_default_events(entities, one_hot)

    return {'id': s_id, 'text': text, 'tokens': tokens,
            'pos_tags': pos_tags,
            'ner_tags': ner_tags, 'entities': entities,
            'event_triggers': event_triggers, 'event_roles': event_roles}


def get_events(relations, one_hot):
    """
    Only creates event_triggers and event_roles for relation mentions in document

    :param relations: relation mentions in document
    :param one_hot: whether to one hot encode labels
    :return: event_triggers and event_roles for relation mentions in document
    """
    event_triggers = []
    event_roles = []

    filtered_relations = [
        rm for rm in relations if rm['name'] in SD4M_RELATION_TYPES
    ]
    for rm in filtered_relations:
        # collect trigger(s) in relation mention
        trigger = next((arg for arg in rm['args'] if arg['role'] == 'trigger'), None)
        if trigger is None or trigger['conceptMention']['type'] not in ['TRIGGER', 'trigger']:
            print('Skipping invalid event')
            continue
        trigger_id = trigger['conceptMention']['id']
        event_trigger = {'id': trigger_id}
        if one_hot:
            event_trigger['event_type_probs'] = encode.one_hot_encode(rm['name'], SD4M_RELATION_TYPES)
        else:
            event_trigger['event_type'] = rm['name']
        event_triggers.append(event_trigger)

        args = [arg for arg in rm['args'] if arg['role'] != 'trigger']
        for arg in args:
            event_role = {
                'trigger': trigger_id,
                'argument': arg['conceptMention']['id'],
            }
            if one_hot:
                event_role['event_argument_probs'] = encode.one_hot_encode(arg['role'], ROLE_LABELS)
            else:
                event_role['event_argument'] = arg['role']
            event_roles.append(event_role)

    return event_triggers, event_roles


def build_default_events(entities, one_hot):
    """
    Builds event triggers for every entity of type 'trigger' and event roles for every trigger-entity pair
    with default label (negative trigger/ role label)

    :param entities: Concept mentions in document
    :param one_hot: Whether to one hot encode labels
    :return: All possible event_triggers and event_roles with default labels
    """
    event_triggers = []
    event_roles = []
    # I initially set the fillers to None in case the document did not have relationMentions
    if one_hot:
        trigger_filler = encode.one_hot_encode(NEGATIVE_TRIGGER_LABEL, SD4M_RELATION_TYPES)
        arg_role_filler = encode.one_hot_encode(NEGATIVE_ARGUMENT_LABEL, ROLE_LABELS)
    else:
        trigger_filler = NEGATIVE_TRIGGER_LABEL
        arg_role_filler = NEGATIVE_ARGUMENT_LABEL

    # Set up all possible triggers with default trigger label
    for entity in entities:
        if entity['entity_type'] in ['TRIGGER', 'trigger']:
            event_trigger = {'id': entity['id']}
            if one_hot:
                event_trigger['event_type_probs'] = encode.one_hot_encode(trigger_filler, SD4M_RELATION_TYPES)
            else:
                event_trigger['event_type'] = trigger_filler
            event_triggers.append(event_trigger)

    # Build all possible pairs of triggers and entities with default argument role label
    for trigger in event_triggers:
        for entity in entities:
            if trigger['id'] != entity['id']:
                event_role = {
                    'trigger': trigger['id'],
                    'argument': entity['id']
                }
                if one_hot:
                    event_role['event_argument_probs'] = arg_role_filler
                else:
                    event_role['event_argument'] = arg_role_filler
                event_roles.append(event_role)

    return event_triggers, event_roles


def update_events(event_triggers, event_roles, relations, one_hot):
    """
    Assumes preceding build_defaults_events step and uses relationMentions of the document
    to update the event_type/ event_role attributes.

    :param event_triggers: Entities of type 'trigger'
    :param event_roles: trigger-entity pairs
    :param relations: relation mentions in document
    :param one_hot: whether to one hot encode labels
    :return: updated event_triggers and event_roles
    """
    filtered_relations = [
        rm for rm in relations if rm['name'] in SD4M_RELATION_TYPES
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
                event_triggers[trigger_idx]['event_type_probs'] = encode.one_hot_encode(
                    rm['name'],
                    SD4M_RELATION_TYPES
                )
            else:
                event_triggers[trigger_idx]['event_type'] = rm['name']
        except Exception as e:
            print(f'{e}\n Did not find {trigger_id} in: {event_triggers}.')
            continue

        role_args = [arg for arg in rm['args'] if arg['role'] != 'trigger']
        for event_role in event_roles:
            if event_role['trigger'] == trigger_id:
                arg_role = next((arg['role'] for arg in role_args
                                 if arg['conceptMention']['id'] == event_role['argument']),
                                NEGATIVE_ARGUMENT_LABEL)
                if one_hot:
                    event_role['event_argument_probs'] = encode.one_hot_encode(
                        arg_role,
                        ROLE_LABELS
                    )
                else:
                    event_role['event_argument'] = arg_role

    return event_triggers, event_roles


def _convert_ner_tags(ner_tags):
    return [ner_tag[:2] + 'TRIGGER' if ner_tag != 'O' and ner_tag[2:] == 'DISASTER_TYPE' else ner_tag
            for ner_tag in ner_tags]


def convert_entity(text, tokens, entity):
    try:
        start_token_idx, end_token_idx = span_to_token_idxs(tokens, entity['span'])
        # The disaster type relations are not annotated with a trigger but with a disaster type.
        # However they can be used interchangeably, thus convert disaster types to triggers.
        entity_type = entity['type'].lower()
        if entity_type == 'disaster-type':
            entity_type = 'trigger'
        elif entity_type == 'disaster_type':
            entity_type = 'trigger'
        return {
            'id': entity['id'],
            'text': span_to_text(text, entity['span']),
            'entity_type': entity_type,
            'start': start_token_idx,
            'end': end_token_idx + 1,  # Generate exclusive token spans
            'char_start': entity['span']['start'],
            'char_end': entity['span']['end']
        }
    except StopIteration:
        print('Token offset issue')
        print(tokens, '\n', entity)
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


def _is_smart_data_doc(doc):
    return 'fileName' in doc['refids'] and doc['refids']['fileName'].startswith('smartdata')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smartdata avro schema to jsonl converter')
    parser.add_argument('--input', type=str, help='Directory of avro file structure')
    parser.add_argument("--one_hot", action="store_true", help='Do not one hot encode labels')
    parser.add_argument("--no_one_hot", dest='one_hot', action="store_false", help='Do not one hot encode labels')
    parser.add_argument("--build_default_events", action="store_true", help='Build default events and update them')
    parser.add_argument("--no_default_events", dest='build_default_events', action="store_false",
                        help='Build default events and update them')
    parser.add_argument('-f', dest='force_output', action='store_true',
                        help='Force creation of new output folder')
    parser.set_defaults(feature=True)
    arguments = parser.parse_args()
    main(arguments)
