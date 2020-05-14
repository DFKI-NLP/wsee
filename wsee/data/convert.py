import argparse
from pathlib import Path

import pandas as pd
import logging
from fastavro import reader
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from fuzzywuzzy import fuzz

from wsee import NEGATIVE_ARGUMENT_LABEL, NEGATIVE_TRIGGER_LABEL, SD4M_RELATION_TYPES, ROLE_LABELS, SDW_RELATION_TYPES
from wsee.utils import encode

logging.basicConfig(level=logging.INFO)
check_counter: int = 0
fix_counter: int = 0
twitter_docs: Dict[str, int] = {
    'docs': 0,
    'relations': 0,
    'docs_with_relations': 0,
    'sdw_relations': 0,
    'sd4m_relations': 0,
    'docs_with_sdw_relations': 0,
    'docs_with_sd4m_relations': 0
}
rss_docs: Dict[str, int] = {
    'docs': 0,
    'relations': 0,
    'docs_with_relations': 0,
    'sdw_relations': 0,
    'sd4m_relations': 0,
    'docs_with_sdw_relations': 0,
    'docs_with_sd4m_relations': 0
}


def main(args):
    input_path = Path(args.input)
    assert input_path.exists(), 'Input not found: %s'.format(args.input)

    one_hot = args.one_hot
    build_defaults = args.build_default_events
    use_first_trigger = args.use_first_trigger
    convert_event_cause = args.convert_event_cause

    logging.info(f"Reading input from: {input_path}")
    logging.info(f"With settings: one_hot ({one_hot}), build_defaults ({build_defaults}), "
                 f"use_first_trigger ({use_first_trigger}), convert_event_cause ({convert_event_cause})")

    daystream = []
    for split in ['train', 'dev', 'test']:
        logging.info(f'Working on {split}-split')
        input_file = input_path.joinpath(split, f'{split}.avro')
        if build_defaults:
            output_file = input_path.joinpath(split, f'{split}_with_events_and_defaults.jsonl')
        else:
            output_file = input_path.joinpath(split, f'{split}_with_events.jsonl')
        smart_data, daystream_part = convert_file(input_file, one_hot, build_defaults,
                                                  use_first_trigger, convert_event_cause)
        smart_data.to_json(output_file, orient='records', lines=True, force_ascii=False)
        daystream.append(daystream_part)
        global check_counter
        global fix_counter
        logging.info(f"Found {check_counter} entity span issues (running sum)")
        logging.info(f"Fixed {fix_counter} entity span issues (running sum)")
    daystream = pd.concat(daystream, sort=False).reset_index(drop=True)
    daystream_output_file = input_path.joinpath('daystream.jsonl')
    daystream.to_json(daystream_output_file, orient='records', lines=True, force_ascii=False)
    global twitter_docs
    global rss_docs
    print(f"Twitter: {twitter_docs}")
    print(f"RSS: {rss_docs}")


def convert_file(file_path, one_hot=False, build_defaults=False, use_first_trigger: bool = False,
                 convert_event_cause: bool = False):
    sd_list = []
    daystream_list = []
    with open(file_path, 'rb') as input_file:
        avro_reader = reader(input_file)
        for doc in tqdm(avro_reader):
            converted_doc = convert_doc(doc=doc, one_hot=one_hot, build_defaults=build_defaults,
                                        use_first_trigger=use_first_trigger, convert_event_cause=convert_event_cause)
            if converted_doc:
                if _is_smart_data_doc(doc):
                    sd_list.append(converted_doc)
                else:
                    daystream_list.append(converted_doc)
    smart_data = pd.DataFrame(sd_list)
    daystream = pd.DataFrame(daystream_list)
    return smart_data, daystream


def convert_doc(doc: Dict, doc_text: str = None, one_hot=False, build_defaults: bool = False,
                use_first_trigger: bool = False, convert_event_cause: bool = False):
    if doc_text is None:
        try:
            doc_text = str(doc['text'])
        except KeyError:
            logging.info(doc['id'], "gave KeyError for doc['text'] and doc_text was None.")
            return None
    s_id = doc['id']
    text = span_to_text(doc_text, doc['span']) if 'span' in doc else doc_text
    tokens = [span_to_text(doc_text, token['span']) for token in
              doc['tokens']]
    ner_tags = _convert_ner_tags([token['ner'] for token in doc['tokens']], convert_event_cause)
    pos_tags = [token['posTag'] for token in doc['tokens']]

    entities = []
    for cm in doc['conceptMentions']:
        converted_entity = convert_entity(text=doc_text, tokens=doc['tokens'], entity=cm,
                                          convert_event_cause=convert_event_cause, doc_id=s_id)
        if converted_entity:
            entities.append(converted_entity)

    event_triggers = []
    event_roles = []
    if _is_smart_data_doc(doc):
        global twitter_docs
        global rss_docs
        if doc['docType'] == 'TWITTER_JSON':
            twitter_docs['docs'] += 1
        elif doc['docType'] == 'RSS_XML':
            rss_docs['docs'] += 1
        if doc['relationMentions']:
            sdw_relations = [
                rm for rm in doc['relationMentions'] if rm['name'] in SDW_RELATION_TYPES
            ]
            sd4m_relations = [
                rm for rm in doc['relationMentions'] if rm['name'] in SD4M_RELATION_TYPES
            ]
            if doc['docType'] == 'TWITTER_JSON':
                twitter_docs['relations'] += len(doc['relationMentions'])
                if len(sdw_relations) > 0 or len(sd4m_relations) > 0:
                    twitter_docs['docs_with_relations'] += 1
                twitter_docs['sdw_relations'] += len(sdw_relations)
                twitter_docs['sd4m_relations'] += len(sd4m_relations)
                if len(sdw_relations) > 0:
                    twitter_docs['docs_with_sdw_relations'] += 1
                if len(sd4m_relations) > 0:
                    twitter_docs['docs_with_sd4m_relations'] += 1
            elif doc['docType'] == 'RSS_XML':
                rss_docs['relations'] += len(doc['relationMentions'])
                if len(sdw_relations) > 0 or len(sd4m_relations) > 0:
                    rss_docs['docs_with_relations'] += 1
                rss_docs['sdw_relations'] += len(sdw_relations)
                rss_docs['sd4m_relations'] += len(sd4m_relations)
                if len(sdw_relations) > 0:
                    rss_docs['docs_with_sdw_relations'] += 1
                if len(sd4m_relations) > 0:
                    rss_docs['docs_with_sd4m_relations'] += 1
            if build_defaults:
                # 1. Does it make sense to build default events and update them, i.e. create negative examples?
                event_triggers, event_roles = build_default_events(entities, one_hot)
                event_triggers, event_roles = update_events(event_triggers, event_roles, doc['relationMentions'],
                                                            one_hot, use_first_trigger)
            else:
                # 2. Or does it make more sense to only create events for corresponding relation mentions in doc?
                event_triggers, event_roles = get_events(doc['relationMentions'], one_hot, use_first_trigger)
    else:
        # Daystream documents do not have relation mention annotation
        event_triggers, event_roles = build_default_events(entities, one_hot)

    return {'id': s_id, 'text': text, 'tokens': tokens, 'docType': doc['docType'],
            'pos_tags': pos_tags, 'ner_tags': ner_tags, 'entities': entities,
            'event_triggers': event_triggers, 'event_roles': event_roles}


def get_events(relations, one_hot, use_first_trigger: bool = False):
    """
    Only creates event_triggers and event_roles for relation mentions in document

    :param use_first_trigger:
    :param relations: relation mentions in document
    :param one_hot: whether to one hot encode labels
    :return: event_triggers and event_roles for relation mentions in document
    """
    event_triggers = []
    event_roles = []

    sd4m_relations = [
        rm for rm in relations if rm['name'] in SD4M_RELATION_TYPES
    ]
    for rm in sd4m_relations:
        # collect trigger(s) in relation mention
        triggers = [arg for arg in rm['args'] if arg['role'] == 'trigger']
        if not triggers:
            logging.info(f'Skipping invalid event: {rm}')
            continue
        if use_first_trigger:
            # if there is a "aus" in triggers choose the "fallen"/"fällt"
            if len(triggers) > 1:
                aus_trigger = next(
                    (trigger for trigger in triggers if trigger['conceptMention']['normalizedValue'] == 'aus'), None)
                fallen_trigger = next(
                    (trigger for trigger in triggers
                     if trigger['conceptMention']['normalizedValue'] in ['fällt', 'faellt', 'fallen']), None)
                if aus_trigger and fallen_trigger:
                    triggers = [fallen_trigger]
                else:
                    triggers = triggers[0:1]
        for trigger in triggers:
            trigger_id = trigger['conceptMention']['id']
            event_trigger = {'id': trigger_id}
            if one_hot:
                event_trigger['event_type_probs'] = encode.one_hot_encode(rm['name'], SD4M_RELATION_TYPES)
            else:
                event_trigger['event_type'] = rm['name']
            event_triggers.append(event_trigger)

            args = [arg for arg in rm['args'] if arg['role'] != 'trigger' and arg['conceptMention']['id'] != trigger_id]
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


def update_events(event_triggers, event_roles, relations, one_hot: bool = True, use_first_trigger: bool = False):
    """
    Assumes preceding build_defaults_events step and uses relationMentions of the document
    to update the event_type/ event_role attributes.

    :param use_first_trigger: Only use the first trigger in relation mention arguments.
    :param event_triggers: Entities of type 'trigger'
    :param event_roles: trigger-entity pairs
    :param relations: relation mentions in document
    :param one_hot: whether to one hot encode labels
    :return: updated event_triggers and event_roles
    """
    sd4m_relations = [
        rm for rm in relations if rm['name'] in SD4M_RELATION_TYPES
    ]

    for rm in sd4m_relations:
        # collect trigger(s) in relation mention
        triggers = [arg for arg in rm['args'] if arg['role'] == 'trigger']
        if not triggers:
            logging.info(f'Skipping invalid event: {rm}')
            continue
        if use_first_trigger:
            # if there is a "aus" in triggers choose the "fallen"/"fällt"
            if len(triggers) > 1:
                aus_trigger = next(
                    (trigger for trigger in triggers if trigger['conceptMention']['normalizedValue'] == 'aus'), None)
                fallen_trigger = next(
                    (trigger for trigger in triggers
                     if trigger['conceptMention']['normalizedValue'] in ['fällt', 'faellt', 'fallen']), None)
                if aus_trigger and fallen_trigger:
                    triggers = [fallen_trigger]
                else:
                    triggers = triggers[0:1]
        for trigger in triggers:
            if trigger['conceptMention']['type'] not in ['TRIGGER', 'trigger',
                                                         'EVENT_CAUSE', 'event-cause', 'event_cause',
                                                         'disaster_type', 'disaster-type', 'DISASTER_TYPE']:
                logging.info(f'Skipping invalid event: {rm}')
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
                logging.info(f'{e}\n Did not find {trigger_id} in: {event_triggers}.')
                continue

            role_args = [arg for arg in rm['args'] if arg['role'] != 'trigger']
            for event_role in event_roles:
                if event_role['trigger'] == trigger_id:
                    arg_role = next((arg['role'] for arg in role_args
                                     if arg['conceptMention']['id'] == event_role['argument']),
                                    NEGATIVE_ARGUMENT_LABEL)
                    if arg_role != NEGATIVE_ARGUMENT_LABEL:
                        if one_hot:
                            event_role['event_argument_probs'] = encode.one_hot_encode(
                                arg_role,
                                ROLE_LABELS
                            )
                        else:
                            event_role['event_argument'] = arg_role

    return event_triggers, event_roles


def _convert_ner_tags(ner_tags: List[str], convert_event_cause: bool = False):
    """
    Replaces legacy ner tag DISASTER_TYPE with TRIGGER.
    :param ner_tags: List of ner tags for the document tokens in BIO format
    :param convert_event_cause: Whether to replace EVENT_CAUSE with TRIGGER
    :return: Updated list of ner tags
    """
    if convert_event_cause:
        return [ner_tag[:2] + 'TRIGGER' if ner_tag != 'O' and ner_tag[2:] in ['DISASTER_TYPE', 'EVENT_CAUSE']
                else ner_tag
                for ner_tag in ner_tags]
    else:
        return [ner_tag[:2] + 'TRIGGER' if ner_tag != 'O' and ner_tag[2:] == 'DISASTER_TYPE'
                else ner_tag
                for ner_tag in ner_tags]


def convert_entity(text: str, tokens: List[str], entity: Dict[str, Any], convert_event_cause: bool = False,
                   doc_id: str = None):
    """
    Takes avro entity and converts it into a more compact dictionary representation.
    :param doc_id: ID of the document for debugging
    :param text: Document text
    :param tokens: Document tokens
    :param entity: Avro entity as dictionary
    :param convert_event_cause: Whether to replace the entity type event_cause with trigger
    :return: Compact entity dictionary representation
    """
    try:
        start_token_idx, end_token_idx = span_to_token_idxs(tokens, entity['span'])
        # The disaster type relations are not annotated with a trigger but with a disaster type.
        # However they can be used interchangeably, thus convert disaster types to triggers.
        entity_type = entity['type'].lower()
        if entity_type in ['disaster-type', 'disaster_type']:
            entity_type = 'trigger'
        elif convert_event_cause and entity_type in ['event-cause', 'event_cause']:
            entity_type = 'trigger'
        converted_entity = {
            'id': entity['id'],
            'text': span_to_text(text, entity['span']),
            'entity_type': entity_type,
            'start': start_token_idx,
            'end': end_token_idx + 1,  # Generate exclusive token spans
            'char_start': entity['span']['start'],
            'char_end': entity['span']['end']
        }
        if 'normalizedValue' in entity and len(converted_entity['text']) > 5:
            fuzz_ratio = fuzz.ratio(converted_entity['text'].replace(' ', ''),
                                    entity['normalizedValue'].replace(' ', ''))
            if fuzz_ratio < 90 and entity['normalizedValue'] not in converted_entity['text']:
                entity_span: Dict[str, Any] = entity['span']
                global check_counter
                check_counter += 1
                if doc_id:
                    logging.debug(f"Check document {doc_id} for entity span issue:")
                else:
                    logging.debug("Entity span issue:")
                logging.debug(f"{converted_entity['text']} (text[{entity_span['start']}:{entity_span['end']}]) vs. "
                              f"{entity['normalizedValue']} (entity['normalizedValue'])")
                if len(entity['normalizedValue'])+1 >= len(converted_entity['text']):
                    # +1 because hashtags are sometimes removed in normalizedValue
                    successful_fix, fixed_entity_span = fix_entity_spans(text, entity_span, entity['normalizedValue'])
                    if successful_fix:
                        converted_entity['text'] = span_to_text(text, fixed_entity_span)
                        converted_entity['char_start'] = fixed_entity_span['start']
                        converted_entity['char_end'] = fixed_entity_span['end']
                        logging.debug(f"Fixed entity spans from: "
                                      f"{span_to_text(text, entity['span'])} "
                                      f"(text[{entity_span['start']}:{entity_span['end']}]) to"
                                      f"{converted_entity['text']} "
                                      f"(text[{fixed_entity_span['start']}:{fixed_entity_span['end']}])")
                        global fix_counter
                        fix_counter += 1
                else:
                    logging.debug(f"Normalized value is substring of entity text")
                    logging.debug(f"Difference in length:"
                                  f"{len(entity['normalizedValue'].replace(' ', ''))} (normalizedValue) vs."
                                  f"{len(converted_entity['text'].replace(' ', ''))} (entity_text)")
                    logging.debug(f"Check anyways")
        return converted_entity
    except StopIteration:
        logging.warning('Token offset issue')
        logging.warning(tokens, '\n', entity)
        return None


def span_to_text(text: str, span: Dict[str, Any]):
    return text[span['start']:span['end']]


def fix_entity_spans(text: str, entity_span: Dict[str, Any], normalizedValue: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Compares normalizedValue with text substring acquired via entity spans and tries to fix the spans.
    This is done by shifting the entity spans to the left and looking at the fuzz.ratio.
    It stops if the fuzz.ratio gets worse. A fuzz.ratio of below 97 is treated as an unsuccessful attempt.
    :param text: Document text.
    :param entity_span: Character span of the entity.
    :param normalizedValue: Joined tokens of the entity.
    :return: Tuple of boolean, that indicates whether the attempt was successful, and the (fixed) entity span.
    """
    original_entity_text = span_to_text(text, entity_span)
    shifted_entity_span = {}
    tmp_fuzz_ratio = 0
    tmp_entity_span = {
        'start': entity_span['start'],
        'end': entity_span['end']
    }
    tmp_text = original_entity_text
    for shift in range(1, len(normalizedValue)):
        shifted_entity_span['start'] = entity_span['start'] - shift
        shifted_entity_span['end'] = entity_span['end'] - shift
        if shifted_entity_span['start'] < 0:
            break
        entity_text = span_to_text(text, shifted_entity_span)
        fuzz_ratio = fuzz.ratio(entity_text.replace(' ', ''),
                                normalizedValue.replace(' ', ''))
        if fuzz_ratio == 100:
            return True, shifted_entity_span
        elif fuzz_ratio < tmp_fuzz_ratio:
            if tmp_fuzz_ratio > 97:
                return True, tmp_entity_span
            else:
                break
        tmp_fuzz_ratio = fuzz_ratio
        tmp_text = entity_text
        tmp_entity_span['start'] = shifted_entity_span['start']
        tmp_entity_span['end'] = shifted_entity_span['end']
    logging.info(f"Unable to fix entity_span for {original_entity_text}. Last try with shifted entity spans: "
                 f"{tmp_text} vs. {normalizedValue} (normalized value)")
    return False, entity_span


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
    parser.add_argument("--use_first_trigger", action="store_true",
                        help='Only use first/ most expressive trigger argument in relation')
    parser.add_argument("--convert_event_cause", action="store_true",
                        help='Convert EVENT_CAUSE NER tag and event_cause entity type to TRIGGER/ trigger')
    parser.add_argument('-f', dest='force_output', action='store_true',
                        help='Force creation of new output folder')
    parser.set_defaults(feature=True)
    arguments = parser.parse_args()
    main(arguments)
