import re
import logging
import numpy as np

from typing import Dict, List, Optional, Any, Tuple
from somajo import SoMaJo
from somajo.token import Token
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


def get_entity_idx(entity_id: str, entities: List[Dict[str, Any]]) -> int:
    entity_idx: int = next((idx for idx, x in enumerate(entities) if x['id'] == entity_id), -1)
    if entity_idx < 0:
        raise Exception(f'The entity_id {entity_id} was not found in:\n {entities}')
    else:
        return entity_idx


def get_entity(entity_id: str, entities: List[Dict[str, Any]]) -> Dict:
    entity: Optional[Dict] = next((x for x in entities if x['id'] == entity_id), None)
    if entity:
        return entity
    else:
        raise Exception(f'The entity_id {entity_id} was not found in:\n {entities}')


def get_left_neighbor_entity(entity: Dict[str, Any], entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    entities.sort(key=lambda e: e['start'])
    entity_idx: int = next(idx for idx, item in enumerate(entities) if item['id'] == entity['id'])
    if entity_idx - 1 >= 0:
        return entities[entity_idx - 1]
    else:
        return None


def get_right_neighbor_entity(entity, entities):
    entities.sort(key=lambda e: e['start'])
    entity_idx: int = next(idx for idx, item in enumerate(entities) if item['id'] == entity['id'])
    if entity_idx + 1 < len(entities):
        return entities[entity_idx + 1]
    else:
        return None


@preprocessor()
def pre_trigger_idx(cand: DataPoint) -> DataPoint:
    trigger: Dict[str, Any] = cand.trigger
    trigger_idx: int = get_entity_idx(trigger['id'], cand.entities)
    cand['trigger_idx']: int = trigger_idx
    return cand


@preprocessor()
def pre_argument_idx(cand: DataPoint) -> DataPoint:
    argument: Dict[str, Any] = cand.argument
    argument_idx: int = get_entity_idx(argument['id'], cand.entities)
    cand['argument_idx']: int = argument_idx
    return cand


@preprocessor()
def pre_trigger_left_tokens(cand: DataPoint) -> DataPoint:
    trigger: Dict[str, Any] = cand.trigger
    cand['trigger_left_tokens']: List[str] = get_windowed_left_tokens(trigger, cand.tokens)
    return cand


@preprocessor()
def pre_trigger_left_pos(cand: DataPoint) -> DataPoint:
    trigger: Dict[str, Any] = cand.trigger
    if 'pos' in cand:
        cand['trigger_left_pos']: List[str] = get_windowed_left_pos(trigger, cand.pos)
    return cand


@preprocessor()
def pre_trigger_left_ner(cand: DataPoint) -> DataPoint:
    trigger: Dict[str, Any] = cand.trigger
    if 'ner' in cand:
        cand['trigger_left_ner']: List[str] = get_windowed_left_ner(trigger, cand.ner_tags)
    return cand


@preprocessor()
def pre_argument_left_tokens(cand: DataPoint) -> DataPoint:
    argument: Dict[str, Any] = cand.argument
    cand['argument_left_tokens']: List[str] = get_windowed_left_tokens(argument, cand.tokens)
    return cand


@preprocessor()
def pre_argument_left_pos(cand: DataPoint) -> DataPoint:
    argument: Dict[str, Any] = cand.argument
    if 'pos' in cand:
        cand['argument_left_pos']: List[str] = get_windowed_left_pos(argument, cand.pos)
    return cand


@preprocessor()
def pre_argument_left_ner(cand: DataPoint) -> DataPoint:
    argument: Dict[str, Any] = cand.argument
    if 'ner' in cand:
        cand['argument_left_ner']: List[str] = get_windowed_left_ner(argument, cand.ner_tags)
    return cand


def get_windowed_left_tokens(entity: Dict, tokens: List[str], window_size: int = None) -> List[str]:
    window_end: int = entity['start']
    if window_size is None:
        window_start: int = 0
    else:
        window_start: int = max(0, window_end - window_size)

    return tokens[window_start:window_end]


def get_windowed_left_pos(entity: Dict, pos: List[str], window_size: int = None) -> List[str]:
    return get_windowed_left_tokens(entity, pos, window_size)


def get_windowed_left_ner(entity: Dict, ner: List[str], window_size: int = None) -> List[str]:
    return get_windowed_left_tokens(entity, ner, window_size)


@preprocessor()
def pre_trigger_right_tokens(cand: DataPoint) -> DataPoint:
    trigger: Dict[str, Any] = cand.trigger
    cand['trigger_right_tokens']: List[str] = get_windowed_right_tokens(trigger, cand.tokens)
    return cand


@preprocessor()
def pre_trigger_right_pos(cand: DataPoint) -> DataPoint:
    trigger: Dict[str, Any] = cand.trigger
    if 'pos' in cand:
        cand['trigger_right_pos']: List[str] = get_windowed_right_pos(trigger, cand.pos)
    return cand


@preprocessor()
def pre_trigger_right_ner(cand: DataPoint) -> DataPoint:
    trigger: Dict[str, Any] = cand.trigger
    if 'ner' in cand:
        cand['trigger_right_ner']: List[str] = get_windowed_right_ner(trigger, cand.ner_tags)
    return cand


@preprocessor()
def pre_argument_right_tokens(cand: DataPoint) -> DataPoint:
    argument: Dict[str, Any] = cand.argument
    cand['argument_right_tokens']: List[str] = get_windowed_right_tokens(argument, cand.tokens)
    return cand


@preprocessor()
def pre_argument_right_pos(cand: DataPoint) -> DataPoint:
    argument: Dict[str, Any] = cand.argument
    if 'pos' in cand:
        cand['argument_right_pos']: List[str] = get_windowed_right_pos(argument, cand.pos)
    return cand


@preprocessor()
def pre_argument_right_ner(cand: DataPoint) -> DataPoint:
    argument: Dict[str, Any] = cand.argument
    if 'ner' in cand:
        cand['argument_right_ner']: List[str] = get_windowed_right_ner(argument, cand.ner_tags)
    return cand


def get_windowed_right_tokens(entity: Dict, tokens: List[str], window_size: int = None) -> List[str]:
    window_start: int = entity['end']
    if window_size is None:
        window_end: int = len(tokens)
    else:
        window_end: int = min(len(tokens), window_start + window_size)

    return tokens[window_start:window_end]


def get_windowed_right_pos(entity: Dict, pos: List[str], window_size: int = None) -> List[str]:
    return get_windowed_right_tokens(entity, pos, window_size)


def get_windowed_right_ner(entity: Dict, ner: List[str], window_size: int = None) -> List[str]:
    return get_windowed_right_tokens(entity, ner, window_size)


@preprocessor()
def pre_between_tokens(cand: DataPoint) -> DataPoint:
    cand['between_tokens']: List[str] = get_between_tokens(cand)
    return cand


def get_between_tokens(cand: DataPoint) -> List[str]:
    trigger: Dict[str, Any] = cand.trigger
    argument: Dict[str, Any] = cand.argument
    if trigger['end'] <= argument['start']:
        start: int = trigger['end']
        end: int = argument['start']
    elif argument['end'] <= trigger['start']:
        start: int = argument['end']
        end: int = trigger['start']
    else:
        logging.debug(f"Trigger {trigger['text']}({trigger['start']}, {trigger['end']}) and "
                      f"argument {argument['text']}({argument['start']}, {argument['end']}) are "
                      f"overlapping.")
        return []

    return cand.tokens[start:end]


@preprocessor()
def pre_between_distance(cand: DataPoint) -> DataPoint:
    cand['between_distance']: int = get_between_distance(cand)
    return cand


def get_between_distance(cand: DataPoint) -> int:
    trigger: Dict[str, Any] = cand.trigger
    argument: Dict[str, Any] = cand.argument
    return get_entity_distance(trigger, argument)


@preprocessor()
def pre_all_trigger_distances(cand: DataPoint) -> DataPoint:
    cand['all_trigger_distances']: Dict[str, Any] = get_all_trigger_distances(cand)
    return cand


def get_all_trigger_distances(cand: DataPoint) -> Dict[str, int]:
    argument: Dict[str, Any] = cand.argument
    all_trigger_distances: Dict[str, int] = {}
    event_triggers: List[Dict] = cand.event_triggers
    for event_trigger in event_triggers:
        trigger_id = event_trigger['id']
        if trigger_id != argument['id']:
            trigger: Dict[str, Any] = get_entity(trigger_id, cand.entities)
            distance: int = get_entity_distance(trigger, argument)
            all_trigger_distances[trigger_id]: int = distance
    return all_trigger_distances


@preprocessor()
def pre_entity_trigger_distances(cand: DataPoint) -> DataPoint:
    cand['entity_trigger_distances']: Dict[str, List[int]] = get_entity_trigger_distances(cand)
    return cand


def get_entity_trigger_distances(cand: DataPoint) -> Dict[str, List[int]]:
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
    trigger: Dict[str, Any] = cand.trigger
    entities: List[Dict] = cand.entities
    for entity in entities:
        if entity['id'] != trigger['id']:
            distance: int = get_entity_distance(trigger, entity)
            entity_type: str = entity['entity_type']
            if entity_type in entity_trigger_distances:
                entity_trigger_distances[entity_type].append(distance)
            else:
                entity_trigger_distances[entity_type]: List[int] = [distance]
    return entity_trigger_distances


def get_closest_entity(cand: DataPoint) -> Optional[Dict]:
    closest_entity: Optional[Dict] = None
    min_distance: int = 10000
    entities: List[Dict] = cand.entities
    trigger: Dict[str, Any] = cand.trigger
    for entity in entities:
        if entity['id'] != trigger['id']:
            distance: int = get_entity_distance(trigger, entity)
            if distance < min_distance:
                closest_entity = entity
                min_distance = distance
    return closest_entity


@preprocessor()
def pre_sentence_trigger_distances(cand: DataPoint) -> DataPoint:
    cand['sentence_trigger_distances']: Dict[str, int] = get_sentence_trigger_distances(cand)
    return cand


def get_sentence_trigger_distances(cand: DataPoint) -> Dict[str, int]:
    """
    Calculates the distances from the argument to all the triggers in the same sentence.
    :param cand:
    :return: Distances from argument to triggers of the same sentence.
    """
    somajo_dictionary: Dict[str, Any] = cand.somajo_doc
    somajo_doc: List[List[Token]] = somajo_dictionary['doc']
    sentences: List[str] = somajo_dictionary['sentences']
    argument: Dict[str, Any] = cand.argument
    somajo_argument: str = somajo_dictionary['entities'][argument['id']]

    sentence_trigger_distances: Dict[str, int] = {}
    event_triggers: List[Dict] = cand.event_triggers
    for event_trigger in event_triggers:
        trigger_id: str = event_trigger['id']
        if trigger_id != argument['id']:
            trigger: Dict[str, Any] = get_entity(trigger_id, cand.entities)

            text: str = ""
            tolerance: int = 0
            for sentence, sent_tokens in zip(sentences, somajo_doc):
                sentence_start = len(text)
                text += sentence
                sentence_end = len(text)
                # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
                # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
                tolerance += 2 * len(
                    [token for token in sent_tokens if token.text in punctuation_marks]) + sentence.count('\n')
                m_start: int = min(trigger['char_start'], argument['char_start'])
                m_end: int = max(trigger['char_end'], argument['char_end'])

                somajo_trigger: str = cand.somajo_doc['entities'][trigger_id]

                if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance and \
                        somajo_trigger in sentence and somajo_argument in sentence:
                    distance: int = get_entity_distance(trigger, argument)
                    sentence_trigger_distances[trigger_id]: int = distance
                    break
    return sentence_trigger_distances


def get_entity_distance(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> int:
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
    cand['entity_type_freqs']: Dict[str, int] = get_entity_type_freqs(cand)
    return cand


def get_entity_type_freqs(cand: DataPoint) -> DataPoint:
    return get_windowed_entity_type_freqs(entities=cand.entities)


def get_windowed_entity_type_freqs(entities: List[Dict[str, Any]], entity: Optional[Dict[str, Any]] = None,
                                   window_size: int = None) -> Dict[str, int]:
    entity_type_freqs: Dict[str, int] = {}

    # Token based start, end numbers
    if entity is not None and window_size is not None:
        window_start: int = max(entity['start'] - window_size, 0)
        window_end: int = min(entity['end'] + window_size, len(entities))
        # Reduce entities list to entities within token based window
        relevant_entity_types: List[str] = [entity['entity_type'] for entity in entities
                                            if entity['start'] >= window_start and entity['end'] <= window_end]
    else:
        relevant_entity_types: List[str] = [entity['entity_type'] for entity in entities]

    for entity_type in relevant_entity_types:
        if entity_type in entity_type_freqs:
            entity_type_freqs[entity_type] += 1
        else:
            entity_type_freqs[entity_type] = 1
    return entity_type_freqs


def check_spans(tokens: List[str], text: str, entity_span: Tuple[int, int], match_span: Tuple[int, int]):
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
    glued_tokens: str = " ".join(tokens[:e_end])
    # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
    # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
    tolerance: int = 2 * len([token for token in tokens[:e_end] if token in punctuation_marks]) + text.count('\n')
    return abs(len(glued_tokens) - m_end) <= tolerance


@preprocessor()
def pre_mixed_ner(cand: DataPoint) -> DataPoint:
    """
    Builds mixed NER patterns from text and entities.
    :param cand:
    :return:
    """
    mixed_ner, mixed_ner_spans = get_mixed_ner(cand)
    cand['mixed_ner']: str = mixed_ner
    cand['mixed_ner_spans']: List[Tuple[int, int]] = mixed_ner_spans
    return cand


def get_mixed_ner(cand: DataPoint) -> (str, List[Tuple[int, int]]):
    """
    Builds mixed NER patterns from text and entities.
    :param cand:
    :return:
    """
    # simple solution with additional token span check
    mixed_ner: str = ''
    offset: int = 0
    mixed_ner_spans: List[Tuple[int, int]] = []
    # TODO: ensure that entities are sorted according to their spans
    entities: List[Dict] = cand.entities
    for idx, entity in enumerate(entities):
        if 'char_start' and 'char_end' in entity.keys():
            match_start: int = entity['char_start']
            match_end: int = entity['char_end']
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


def get_somajo_doc_tokens(doc: List[List[Token]]) -> List[str]:
    return [token.text for sentence in doc for token in sentence]


def get_somajo_doc_sentences(doc: List[List[Token]]):
    # potentially introduces whitespaces that were not in the original document
    return [" ".join([token.text for token in sentence]) for sentence in doc]


@preprocessor()
def pre_somajo_doc(cand: DataPoint) -> DataPoint:
    cand['somajo_doc'] = get_somajo_doc(cand)
    return cand


def get_somajo_doc(cand: DataPoint) -> Dict[str, Any]:
    load_somajo_model()
    somajo_doc: List[List[Token]] = list(nlp_somajo.tokenize_text([cand.text]))
    entities: Dict[str, str] = {}
    original_entities: List[Dict[str, Any]] = cand.entities
    for entity in original_entities:
        tokenized_entity: List[str] = get_somajo_doc_tokens(nlp_somajo.tokenize_text([entity['text']]))
        entities[entity['id']] = (' '.join(tokenized_entity))

    doc = {
        'doc': somajo_doc,
        'tokens': get_somajo_doc_tokens(somajo_doc),
        'sentences': get_somajo_doc_sentences(somajo_doc),
        'entities': entities,
    }
    return doc


def get_somajo_separate_sentence(cand: DataPoint) -> bool:
    assert 'somajo_doc' in cand, 'You need to run get_somajo_doc first and add somajo_doc to the dataframe.'
    if len(cand.somajo_doc['sentences']) == 1:
        return False
    same_sentence: bool = False
    text: str = ""
    tolerance: int = 0
    trigger: Dict[str, Any] = cand.trigger
    argument: Dict[str, Any] = cand.argument
    somajo_doc: Dict[str, Any] = cand.somajo_doc
    for sentence, sent_tokens in zip(somajo_doc['sentences'], somajo_doc['doc']):
        sentence_start: int = len(text)
        text += sentence
        sentence_end: int = len(text)
        # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
        # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
        tolerance += 2 * len([token for token in sent_tokens if token.text in punctuation_marks]) + sentence.count('\n')
        m_start: int = min(trigger['char_start'], argument['char_start'])
        m_end: int = max(trigger['char_end'], argument['char_end'])

        somajo_trigger: str = somajo_doc['entities'][trigger['id']]
        somajo_argument: str = somajo_doc['entities'][argument['id']]

        if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance and \
                somajo_trigger in sentence and somajo_argument in sentence:
            trigger_matches = [m.start() for m in re.finditer(escape_regex_chars(somajo_trigger), sentence)]
            trigger_in_sentence: bool = any(
                abs(trigger_match + sentence_start - trigger['char_start']) < tolerance
                for trigger_match in trigger_matches)
            argument_matches = [m.start() for m in re.finditer(escape_regex_chars(somajo_argument), sentence)]
            argument_in_sentence: bool = any(
                abs(argument_match + sentence_start - argument['char_start']) < tolerance
                for argument_match in argument_matches)
            if trigger_in_sentence and argument_in_sentence:
                same_sentence = True
                break
    if same_sentence:
        return False
    else:
        return True


def get_sentence_entities(cand: DataPoint) -> List[Dict[str, Any]]:
    """
    Returns all the entities that are in the same sentence as the trigger and the argument.
    If the trigger and argument are not in the same sentence, an empty list is returned.
    :param cand: DataPoint
    :return: Same sentence entities.
    """
    assert 'somajo_doc' in cand, "Need somajo_doc to retrieve sentence entities"
    text: str = ""
    tolerance: int = 0
    trigger: Dict[str, Any] = cand.trigger
    argument: Dict[str, Any] = cand.argument
    somajo_doc: Dict[str, Any] = cand.somajo_doc
    entities: List[Dict[str, Any]] = cand.entities
    for sentence, sent_tokens in zip(somajo_doc['sentences'], somajo_doc['doc']):
        sentence_start: int = len(text)
        text += sentence
        sentence_end: int = len(text)
        # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
        # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
        tolerance += 2 * len([token for token in sent_tokens if token.text in punctuation_marks]) + sentence.count('\n')
        m_start: int = min(trigger['char_start'], argument['char_start'])
        m_end: int = max(trigger['char_end'], argument['char_end'])

        somajo_trigger: str = somajo_doc['entities'][trigger['id']]
        somajo_argument: str = somajo_doc['entities'][argument['id']]

        if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance and \
                somajo_trigger in sentence and somajo_argument in sentence:
            return [entity for entity in entities
                    if sentence_start <= entity['char_start'] + tolerance and
                    entity['char_end'] <= sentence_end + tolerance and
                    somajo_trigger in sentence]
    return []


def is_multiple_same_event_type(cand: DataPoint) -> bool:
    between_tokens: List[str] = get_between_tokens(cand)
    trigger: Dict[str, Any] = cand.trigger
    trigger_text: str = trigger['text']
    if trigger_text in between_tokens:
        return True
    else:
        return False


# only for exploration purposes when gold labels are available
def get_event_types(cand: DataPoint) -> DataPoint:
    if 'event_triggers' in cand:
        event_types = []
        for event_trigger in cand.event_triggers:
            entity: Dict[str, Any] = get_entity(event_trigger['id'], cand.entities)
            label: int = np.asarray(event_trigger['event_type_probs']).argmax()
            event_types.append((entity['text'], (entity['char_start'], entity['char_end']), label))
        cand['event_types'] = event_types
    return cand


def get_event_arg_roles(cand: DataPoint) -> DataPoint:
    if 'event_triggers' and 'event_roles' in cand:
        event_arg_roles = []
        for event_role in cand.event_roles:
            role_label = np.asarray(event_role['event_argument_probs']).argmax()
            if role_label != 10:
                trigger: Dict[str, Any] = get_entity(event_role['trigger'], cand.entities)
                event_type = next((np.asarray(event_trigger['event_type_probs']).argmax()
                                   for event_trigger in cand.event_triggers
                                   if event_trigger['id'] == event_role['trigger']), 7)
                argument: Dict[str, Any] = get_entity(event_role['argument'], cand.entities)
                role_label = np.asarray(event_role['event_argument_probs']).argmax()
                event_arg_roles.append(((trigger['text'], (trigger['char_start'], trigger['char_end']), event_type),
                                        (argument['text'], argument['entity_type'],
                                         (argument['char_start'], argument['char_end'])), role_label))
        cand['event_arg_roles'] = event_arg_roles
    return cand
