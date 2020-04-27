import re
import logging
import numpy as np

from typing import Dict, List, Optional, Callable, Any
from somajo import SoMaJo
from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint
from wsee.preprocessors.pattern_event_processor import escape_regex_chars

punctuation_marks = ["<", "(", "[", "{", "\\", "^", "-", "=", "$", "!", "|",
                     "]", "}", ")", "?", "*", "+", ".", ",", ":", ";", ">",
                     "_", "#", "/"]

nlp_somajo: Optional[SoMaJo] = None


def load_somajo_model():
    global nlp_somajo
    if nlp_somajo is None:
        nlp_somajo = SoMaJo("de_CMC", split_camel_case=True)


def get_entity_idx(entity_id: str, entities: List[Dict[str, Any]]):
    entity_idx: int = next((idx for idx, x in enumerate(entities) if x['id'] == entity_id), -1)
    if entity_idx < 0:
        raise Exception(f'The entity_id {entity_id} was not found in:\n {entities}')
    else:
        return entity_idx


def get_entity(entity_id, entities):
    entity = next((x for x in entities if x['id'] == entity_id), None)
    if entity:
        return entity
    else:
        raise Exception(f'The entity_id {entity_id} was not found in:\n {entities}')


def get_entity_text_and_type(entity_id, entities):
    entity = get_entity(entity_id, entities)
    return entity['text'], entity['entity_type']


@preprocessor()
def pre_trigger_idx(cand: DataPoint) -> DataPoint:
    trigger_idx: int = get_entity_idx(cand.trigger['id'], cand.entities)
    cand['trigger_idx'] = trigger_idx
    return cand


@preprocessor()
def pre_argument_idx(cand: DataPoint) -> DataPoint:
    argument_idx: int = get_entity_idx(cand.argument['id'], cand.entities)
    cand['argument_idx'] = argument_idx
    return cand


@preprocessor()
def pre_trigger_left_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger['id'], cand.entities)
    cand['trigger_left_tokens'] = get_windowed_left_tokens(trigger, cand.tokens)
    return cand


@preprocessor()
def pre_trigger_left_pos(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger['id'], cand.entities)
    if 'pos' in cand:
        cand['trigger_left_pos'] = get_windowed_left_pos(trigger, cand.pos)
    return cand


@preprocessor()
def pre_trigger_left_ner(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger['id'], cand.entities)
    if 'ner' in cand:
        cand['trigger_left_ner'] = get_windowed_left_ner(trigger, cand.ner_tags)
    return cand


@preprocessor()
def pre_argument_left_tokens(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument['id'], cand.entities)
    cand['argument_left_tokens'] = get_windowed_left_tokens(argument, cand.tokens)
    return cand


@preprocessor()
def pre_argument_left_pos(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument['id'], cand.entities)
    if 'pos' in cand:
        cand['argument_left_pos'] = get_windowed_left_pos(argument, cand.pos)
    return cand


@preprocessor()
def pre_argument_left_ner(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument['id'], cand.entities)
    if 'ner' in cand:
        cand['argument_left_ner'] = get_windowed_left_ner(argument, cand.ner_tags)
    return cand


def get_windowed_left_tokens(entity, tokens, window_size: int = None) -> List[str]:
    window_end = entity['start']
    if window_size is None:
        window_start = 0
    else:
        window_start = max(0, window_end - window_size)

    return tokens[window_start:window_end]


def get_windowed_left_pos(entity, pos, window_size: int = None) -> List[str]:
    return get_windowed_left_tokens(entity, pos, window_size)


def get_windowed_left_ner(entity, ner, window_size: int = None) -> List[str]:
    return get_windowed_left_tokens(entity, ner, window_size)


@preprocessor()
def pre_trigger_right_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger['id'], cand.entities)
    cand['trigger_right_tokens'] = get_windowed_right_tokens(trigger, cand.tokens)
    return cand


@preprocessor()
def pre_trigger_right_pos(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger['id'], cand.entities)
    if 'pos' in cand:
        cand['trigger_right_pos'] = get_windowed_right_pos(trigger, cand.pos)
    return cand


@preprocessor()
def pre_trigger_right_ner(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger['id'], cand.entities)
    if 'ner' in cand:
        cand['trigger_right_ner'] = get_windowed_right_ner(trigger, cand.ner_tags)
    return cand


@preprocessor()
def pre_argument_right_tokens(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument['id'], cand.entities)
    cand['argument_right_tokens'] = get_windowed_right_tokens(argument, cand.tokens)
    return cand


@preprocessor()
def pre_argument_right_pos(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument['id'], cand.entities)
    if 'pos' in cand:
        cand['argument_right_pos'] = get_windowed_right_pos(argument, cand.pos)
    return cand


@preprocessor()
def pre_argument_right_ner(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument['id'], cand.entities)
    if 'ner' in cand:
        cand['argument_right_ner'] = get_windowed_right_ner(argument, cand.ner_tags)
    return cand


def get_windowed_right_tokens(entity, tokens, window_size: int = None) -> List[str]:
    window_start = entity['end']
    if window_size is None:
        window_end = len(tokens)
    else:
        window_end = min(len(tokens), window_start + window_size)

    return tokens[window_start:window_end]


def get_windowed_right_pos(entity, pos, window_size: int = None) -> List[str]:
    return get_windowed_right_tokens(entity, pos, window_size)


def get_windowed_right_ner(entity, ner, window_size: int = None) -> List[str]:
    return get_windowed_right_tokens(entity, ner, window_size)


@preprocessor()
def pre_between_tokens(cand: DataPoint) -> DataPoint:
    cand['between_tokens'] = get_between_tokens(cand)
    return cand


def get_between_tokens(cand: DataPoint):
    if cand.trigger['end'] <= cand.argument['start']:
        start = cand.trigger['end']
        end = cand.argument['start']
    elif cand.argument['end'] <= cand.trigger['start']:
        start = cand.argument['end']
        end = cand.trigger['start']
    else:
        logging.debug(f"Trigger {cand.trigger['text']}({cand.trigger['start']}, {cand.trigger['end']}) and "
                      f"argument {cand.argument['text']}({cand.argument['start']}, {cand.argument['end']}) are "
                      f"overlapping.")
        cand['between_tokens'] = []
        return cand

    return cand.tokens[start:end]


@preprocessor()
def pre_between_distance(cand: DataPoint) -> DataPoint:
    cand['between_distance'] = get_between_distance(cand)
    return cand


def get_between_distance(cand: DataPoint):
    trigger = get_entity(cand.trigger['id'], cand.entities)
    argument = get_entity(cand.argument['id'], cand.entities)
    return get_entity_distance(trigger, argument)


@preprocessor()
def pre_all_trigger_distances(cand: DataPoint) -> DataPoint:
    cand['all_trigger_distances'] = get_all_trigger_distances(cand)
    return cand


def get_all_trigger_distances(cand: DataPoint):
    argument = get_entity(cand.argument['id'], cand.entities)
    all_trigger_distances = {}
    for event_trigger in cand.event_triggers:
        trigger_id = event_trigger['id']
        if trigger_id != cand.argument['id']:
            trigger = get_entity(trigger_id, cand.entities)
            distance = get_entity_distance(trigger, argument)
            all_trigger_distances[trigger_id] = distance
    return all_trigger_distances


@preprocessor()
def pre_entity_trigger_distances(cand: DataPoint) -> DataPoint:
    cand['entity_trigger_distances'] = get_entity_trigger_distances(cand)
    return cand


def get_entity_trigger_distances(cand: DataPoint):
    """
    Calculates distances from trigger to all other entities, grouped by their entity type.
    :param cand: DataPoint, one example.
    :return: DataPoint enriched with distances.
    """
    entity_trigger_distances: Dict[str, List[int]] = {
        'location': [],
        'location_city': [],
        'location_route': [],
        'location_stop': [],
        'location_street': [],
        'date': [],
        'time': [],
        'duration': [],
        'distance': [],
        'trigger': []
    }
    for entity in cand.entities:
        if entity['id'] != cand.trigger['id']:
            distance = get_entity_distance(cand.trigger, entity)
            entity_type = entity['entity_type']
            if entity_type in entity_trigger_distances:
                entity_trigger_distances[entity_type].append(distance)
            else:
                entity_trigger_distances[entity_type] = [distance]
    return entity_trigger_distances


def get_closest_entity(cand: DataPoint):
    closest_entity = None
    min_distance = 10000
    for entity in cand.entities:
        if entity['id'] != cand.trigger['id']:
            distance = get_entity_distance(cand.trigger, entity)
            if distance < min_distance:
                closest_entity = entity
                min_distance = distance
    return closest_entity


@preprocessor()
def pre_sentence_trigger_distances(cand: DataPoint) -> DataPoint:
    cand['sentence_trigger_distances'] = get_sentence_trigger_distances(cand)
    return cand


def get_sentence_trigger_distances(cand: DataPoint):
    somajo_doc = cand.somajo_doc['doc']
    sentences = cand.somajo_doc['sentences']
    somajo_argument = cand.somajo_doc['entities'][cand.argument['id']]

    sentence_trigger_distances = {}
    for event_trigger in cand.event_triggers:
        trigger_id = event_trigger['id']
        if trigger_id != cand.argument['id']:
            trigger = get_entity(trigger_id, cand.entities)

            text = ""
            tolerance = 0
            for sentence, sent_tokens in zip(sentences, somajo_doc):
                sentence_start = len(text)
                text += sentence
                sentence_end = len(text)
                # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
                # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
                tolerance += 2 * len(
                    [token for token in sent_tokens if token.text in punctuation_marks]) + sentence.count('\n')
                m_start = min(trigger['char_start'], cand.argument['char_start'])
                m_end = max(trigger['char_end'], cand.argument['char_end'])

                somajo_trigger = cand.somajo_doc['entities'][trigger_id]

                if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance and \
                        somajo_trigger in sentence and somajo_argument in sentence:
                    distance = get_entity_distance(trigger, cand.argument)
                    sentence_trigger_distances[trigger_id] = distance
                    break
    return sentence_trigger_distances


def get_entity_distance(entity1, entity2) -> int:
    if entity1['end'] <= entity2['start']:
        return entity2['start'] - entity1['end']
    elif entity2['end'] <= entity1['start']:
        return entity1['start'] - entity2['end']
    else:
        logging.debug(f"Overlapping entities {entity1['id']}:{entity1['text']}({entity1['start']}, {entity1['end']}) "
                      f"and {entity2['id']}:{entity2['text']}({entity2['start']}, {entity2['end']})")
        return -1


@preprocessor()
def pre_entity_type_freqs(cand: DataPoint) -> DataPoint:
    cand['entity_type_freqs'] = get_entity_type_freqs(cand)
    return cand


def get_entity_type_freqs(cand: DataPoint) -> DataPoint:
    return get_windowed_entity_type_freqs(entities=cand.entities)


def get_windowed_entity_type_freqs(entities, entity=None, window_size: int = None) -> Dict[str, int]:
    entity_type_freqs: Dict[str, int] = {}

    # Token based start, end numbers
    if entity is not None and window_size is not None:
        window_start = max(entity['start'] - window_size, 0)
        window_end = min(entity['end'] + window_size, len(entities))
        # Reduce entities list to entities within token based window
        relevant_entity_types = [entity['entity_type'] for entity in entities
                                 if entity['start'] >= window_start and entity['end'] <= window_end]
    else:
        relevant_entity_types = [entity['entity_type'] for entity in entities]

    for entity_type in relevant_entity_types:
        if entity_type in entity_type_freqs:
            entity_type_freqs[entity_type] += 1
        else:
            entity_type_freqs[entity_type] = 1
    return entity_type_freqs


def check_spans(tokens, text, entity_span, match_span):
    """
    Checks token based entity span with character based text match span to see if
    it is plausible for the match to be at that particular position.
        Example: Berliner in Berlin sind besorgt
    If we only look at a text match without checking the span, we may end up replacing the
    first occurrence of "Berlin", resulting in
        LOCATION_CITYer in Berlin sind besorgt
    instead of
        Berliner in LOCATION_CITY sind besorgt
    RegExes could help, but we would have to look at whitespaces, special symbols, etc.
    """
    e_start, e_end = entity_span  # token based
    m_start, m_end = match_span  # char based
    glued_tokens = " ".join(tokens[:e_end])
    # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
    # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
    tolerance = 2 * len([token for token in tokens[:e_end] if token in punctuation_marks]) + text.count('\n')
    return abs(len(glued_tokens) - m_end) <= tolerance


@preprocessor()
def pre_mixed_ner(cand: DataPoint) -> DataPoint:
    """
    Builds mixed NER patterns from text and entities.
    :param cand:
    :return:
    """
    mixed_ner, mixed_ner_spans = get_mixed_ner(cand)
    cand['mixed_ner'] = mixed_ner
    cand['mixed_ner_spans'] = mixed_ner_spans
    return cand


def get_mixed_ner(cand: DataPoint) -> DataPoint:
    """
    Builds mixed NER patterns from text and entities.
    :param cand:
    :return:
    """
    # simple solution with additional token span check
    mixed_ner = ''
    offset = 0
    mixed_ner_spans = []
    # TODO: ensure that entities are sorted according to their spans
    for idx, entity in enumerate(cand.entities):
        if 'char_start' and 'char_end' in entity.keys():
            match_start = entity['char_start']
            match_end = entity['char_end']
            assert cand.text[match_start:match_end] == entity['text'], \
                f"Mismatch {cand.text[match_start:match_end]} {entity['text']} in:\n{cand.text}"
        else:
            # simple text replace with search for entity text + check with token span?
            entity_text = re.compile(escape_regex_chars(entity['text'], optional_hashtag=False))
            matches = list(entity_text.finditer(cand.text))
            relevant_match = next((match for match in matches if check_spans(cand.tokens, cand.text,
                                                                             (entity['start'], entity['end']),
                                                                             (match.start(), match.end()))), None)

            if relevant_match is None:
                print("Something went wrong")
                print(entity)
                mixed_ner = ''
                mixed_ner_spans = []
                break
            else:
                match_start = relevant_match.start()
                match_end = relevant_match.end()
        # TODO: handle new line character differently
        #  replace with whitespace character adjust spans
        mixed_ner += cand.text[offset:match_start] + entity['entity_type'].upper()
        mixed_ner_spans.append((match_start, match_start + len(entity['entity_type'])))
        offset = match_end
    mixed_ner += cand.text[offset:]
    return mixed_ner, mixed_ner_spans


def get_somajo_doc_tokens(doc):
    return [token.text for sentence in doc for token in sentence]


def get_somajo_doc_sentences(doc):
    # introduces whitespaces
    return [" ".join([token.text for token in sentence]) for sentence in doc]


@preprocessor()
def pre_somajo_doc(cand: DataPoint) -> DataPoint:
    cand['somajo_doc'] = get_somajo_doc(cand)
    return cand


def get_somajo_doc(cand: DataPoint):
    load_somajo_model()
    somajo_doc = list(nlp_somajo.tokenize_text([cand.text]))
    entities = {}
    for entity in cand.entities:
        tokenized_entity = get_somajo_doc_tokens(nlp_somajo.tokenize_text([entity['text']]))
        entities[entity['id']] = (' '.join(tokenized_entity))

    doc = {
        'doc': somajo_doc,
        'tokens': get_somajo_doc_tokens(somajo_doc),
        'sentences': get_somajo_doc_sentences(somajo_doc),
        'entities': entities,
    }
    return doc


def get_somajo_separate_sentence(cand: DataPoint):
    assert 'somajo_doc' in cand, 'You need to run get_somajo_doc first and add somajo_doc to the dataframe.'
    if len(cand.somajo_doc['sentences']) == 1:
        return False
    same_sentence = False
    text = ""
    tolerance = 0
    for sentence, sent_tokens in zip(cand.somajo_doc['sentences'], cand.somajo_doc['doc']):
        sentence_start = len(text)
        text += sentence
        sentence_end = len(text)
        # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
        # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
        tolerance += 2 * len([token for token in sent_tokens if token.text in punctuation_marks]) + sentence.count('\n')
        m_start = min(cand.trigger['char_start'], cand.argument['char_start'])
        m_end = max(cand.trigger['char_end'], cand.argument['char_end'])

        somajo_trigger = cand.somajo_doc['entities'][cand.trigger['id']]
        somajo_argument = cand.somajo_doc['entities'][cand.argument['id']]

        if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance and \
                somajo_trigger in sentence and somajo_argument in sentence:
            trigger_matches = [m.start() for m in re.finditer(escape_regex_chars(somajo_trigger), sentence)]
            trigger_in_sentence = any(abs(trigger_match + sentence_start - cand.trigger['char_start']) < tolerance
                                      for trigger_match in trigger_matches)
            argument_matches = [m.start() for m in re.finditer(escape_regex_chars(somajo_argument), sentence)]
            argument_in_sentence = any(abs(argument_match + sentence_start - cand.argument['char_start']) < tolerance
                                       for argument_match in argument_matches)
            if trigger_in_sentence and argument_in_sentence:
                same_sentence = True
                break
    if same_sentence:
        return False
    else:
        return True


def get_sentence_entities(cand: DataPoint):
    """
    Returns all the entities that are in the same sentence as the trigger and the argument.
    If the trigger and argument are not in the same sentence, an empty list is returned.
    :param cand: DataPoint
    :return: Same sentence entities.
    """
    assert 'somajo_doc' in cand, "Need somajo_doc to retrieve sentence entities"
    text = ""
    tolerance = 0
    for sentence, sent_tokens in zip(cand.somajo_doc['sentences'], cand.somajo_doc['doc']):
        sentence_start = len(text)
        text += sentence
        sentence_end = len(text)
        # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
        # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
        tolerance += 2 * len([token for token in sent_tokens if token.text in punctuation_marks]) + sentence.count('\n')
        m_start = min(cand.trigger['char_start'], cand.argument['char_start'])
        m_end = max(cand.trigger['char_end'], cand.argument['char_end'])

        somajo_trigger = cand.somajo_doc['entities'][cand.trigger['id']]
        somajo_argument = cand.somajo_doc['entities'][cand.argument['id']]

        if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance and \
                somajo_trigger in sentence and somajo_argument in sentence:
            return [entity for entity in cand.entities
                    if sentence_start <= entity['char_start'] + tolerance and
                    entity['char_end'] <= sentence_end + tolerance and
                    somajo_trigger in sentence]
    return []


def is_multiple_same_event_type(cand: DataPoint):
    between_tokens = get_between_tokens(cand)
    trigger_text = cand.trigger['text']
    if trigger_text in between_tokens:
        return True
    else:
        return False


# only for exploration purposes when gold labels are available
def get_event_types(cand: DataPoint) -> DataPoint:
    if 'event_triggers' in cand:
        event_types = []
        for event_trigger in cand.event_triggers:
            entity = get_entity(event_trigger['id'], cand.entities)
            label = np.asarray(event_trigger['event_type_probs']).argmax()
            event_types.append((entity['text'], (entity['char_start'], entity['char_end']), label))
        cand['event_types'] = event_types
    return cand


def get_event_arg_roles(cand: DataPoint) -> DataPoint:
    if 'event_triggers' and 'event_roles' in cand:
        event_arg_roles = []
        for event_role in cand.event_roles:
            role_label = np.asarray(event_role['event_argument_probs']).argmax()
            if role_label != 10:
                trigger = get_entity(event_role['trigger'], cand.entities)
                event_type = next((np.asarray(event_trigger['event_type_probs']).argmax()
                                   for event_trigger in cand.event_triggers
                                   if event_trigger['id'] == event_role['trigger']), 7)
                argument = get_entity(event_role['argument'], cand.entities)
                role_label = np.asarray(event_role['event_argument_probs']).argmax()
                event_arg_roles.append(((trigger['text'], (trigger['char_start'], trigger['char_end']), event_type),
                                        (argument['text'], argument['entity_type'],
                                         (argument['char_start'], argument['char_end'])), role_label))
        cand['event_arg_roles'] = event_arg_roles
    return cand
