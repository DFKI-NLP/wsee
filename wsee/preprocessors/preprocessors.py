import re
import logging
import numpy as np

from typing import Dict, List, Optional, Any, Tuple

import spacy
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


def load_spacy_model():
    global nlp_spacy
    if nlp_spacy is None:
        nlp_spacy = spacy.load('de_core_news_md')


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


def get_between_text(cand: DataPoint) -> str:
    min_end: int = min(cand.trigger['char_end'], cand.argument['char_end'])
    max_start: int = max(cand.trigger['char_start'], cand.argument['char_start'])
    between_text: str = cand.text[min_end:max_start]
    return between_text


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
def pre_closest_entity_same_sentence(cand: DataPoint) -> Optional[Dict]:
    cand['closest_entity'] = get_closest_entity_same_sentence(cand)
    return cand


def get_closest_entity_same_sentence(cand: DataPoint) -> Optional[Dict]:
    closest_entity: Optional[Dict] = None
    min_distance: int = 10000
    entities: List[Dict] = cand.entities
    trigger: Dict[str, Any] = cand.trigger
    somajo_dictionary: Dict[str, Any] = cand.somajo_doc
    sentences: List[Dict[str, Any]] = somajo_dictionary['sentences']
    trigger_sentence = {'char_start': 0, 'char_end': len(cand.text)}
    for sentence in sentences:
        if sentence['char_start'] <= trigger['char_start'] and sentence['char_end'] >= trigger['char_end']:
            trigger_sentence['char_start'] = sentence['char_start']
            trigger_sentence['char_end'] = sentence['char_end']
    same_sentence_entities = [entity for entity in entities
                              if entity['char_start'] >= trigger_sentence['char_start'] and
                              entity['char_end'] <= trigger_sentence['char_end']]
    for entity in same_sentence_entities:
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
    sentences: List[Dict[str, Any]] = somajo_dictionary['sentences']
    argument: Dict[str, Any] = cand.argument

    sentence_trigger_distances: Dict[str, int] = {}
    event_triggers: List[Dict] = cand.event_triggers
    for event_trigger in event_triggers:
        trigger_id: str = event_trigger['id']
        if trigger_id != argument['id']:
            trigger: Dict[str, Any] = get_entity(trigger_id, cand.entities)

            tolerance: int = 0
            for sentence in sentences:
                sentence_start = sentence['char_start']
                sentence_end = sentence['char_end']
                m_start: int = min(trigger['char_start'], argument['char_start'])
                m_end: int = max(trigger['char_end'], argument['char_end'])

                if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance:
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


def get_spacy_doc_tokens(doc):
    return [token.text for token in doc]


def get_spacy_doc_sentences(doc):
    return [s.text for s in doc.sents]


def get_spacy_doc(cand: DataPoint):
    load_spacy_model()
    spacy_doc = nlp_spacy(cand.text)
    doc = {
        'doc': spacy_doc,
        'tokens': get_spacy_doc_tokens(spacy_doc),
        'sentences': get_spacy_doc_sentences(spacy_doc),  # preserves sentence texts
        'trigger_text': nlp_spacy(get_entity(cand.trigger['id'], cand.entities)['text']),
    }
    if 'argument_id' in cand:
        doc['argument_text'] = get_entity(cand.argument['id'], cand.entities)['text']
    return doc


@preprocessor()
def pre_spacy_doc(cand: DataPoint) -> DataPoint:
    cand['spacy_doc'] = get_spacy_doc(cand)
    return cand


def get_somajo_doc_tokens(doc: List[List[Token]]) -> List[str]:
    doc_tokens = []
    for sentence in doc:
        for token in sentence:
            if token.original_spelling:
                doc_tokens.append(token.original_spelling)
            else:
                doc_tokens.append(token.text)
    return doc_tokens


def normalize_whitespaces(s):
    """
    Replaces multiple consecutive whitespaces, whitespace characters (\n\t' ') with single whitespace
    :param s: String
    :return: Normalized string
    """
    remove_multiple_ws = re.sub(r"\s\s+", " ", s)
    normalized_ws = re.sub(r"\s", " ", remove_multiple_ws)
    return normalized_ws


def remap_sentence_boundaries(sentence_text: str, sentence_char_start: int, sentence_char_end: int, text: str) \
        -> Tuple[int, int]:
    """
    Uses information from SoMaJo sentence splitting and original text to retrieve correct character offsets for sentence
    :param sentence_text: Reconstructed sentence text from sentence tokens
    :param sentence_char_start: Approximate start char
    :param sentence_char_end: Approximate end char
    :param text: Original document text
    :return: Updated character offsets for sentence
    """
    char_start = sentence_char_start
    char_end = sentence_char_end
    cleaned_text = normalize_whitespaces(text)
    removed_whitespaces = len(text) - len(cleaned_text)

    for i in range(0, removed_whitespaces+1):
        cleaned_sentence = normalize_whitespaces(text[char_start:char_end + i])
        cleaned_text_lstrip = cleaned_sentence.lstrip()
        stripped_leading_ws = len(cleaned_sentence) - len(cleaned_text_lstrip)
        if stripped_leading_ws > 0:
            char_start += stripped_leading_ws
            cleaned_sentence = normalize_whitespaces(text[char_start:char_end + i])
        if sentence_text == cleaned_sentence:
            return char_start, char_end+i
    logging.warning(f"Sentence boundary [A] and text substring [B] do not match, but could not be remapped: "
                    f"\n[A]{sentence_text}\n[B]{text[char_start:char_end]}")
    return char_start, char_end


def get_somajo_doc_sentences(doc: List[List[Token]], text: str) -> List[Dict[str, Any]]:
    """
    Builds sentence dictionaries from SoMaJo sentence splitting and document text
    :param doc: List of sentences, where each sentence is a list of SoMaJo tokens
    :param text: Original document text
    :return: List of sentences with sentence text and spans (character and token level)
    """
    sentences: List[Dict[str, Any]] = []
    char_offset = 0
    token_idx = 0
    if len(doc) > 1:
        for sentence in doc:
            sentence_text = ""
            add_space_after = False  # whether to increase char_offset after sentence end
            sentence_start = token_idx
            sentence_end = sentence_start
            for token in sentence:
                sentence_end += 1
                token_text = token.original_spelling if token.original_spelling else token.text
                sentence_text += token_text
                if token.space_after:
                    if not token.last_in_sentence:
                        sentence_text += " "
                    else:
                        add_space_after = True
            sentence_char_start = char_offset
            sentence_char_end = sentence_char_start + len(sentence_text)
            sentence_char_start, sentence_char_end = remap_sentence_boundaries(sentence_text,
                                                                               sentence_char_start,
                                                                               sentence_char_end, text)
            tmp_sentence = {
                'text': text[sentence_char_start:sentence_char_end],
                'start': sentence_start,
                'end': sentence_end,
                'char_start': sentence_char_start,
                'char_end': sentence_char_end
            }
            sentences.append(tmp_sentence)
            token_idx = sentence_end
            char_offset = sentence_char_end
            if add_space_after:
                char_offset += 1
    else:
        sentences.append({
            'text': text,
            'start': 0,
            'end': len(doc[0]),
            'char_start': 0,
            'char_end': len(text)
        })
    return sentences


@preprocessor()
def pre_somajo_doc(cand: DataPoint) -> DataPoint:
    cand['somajo_doc'] = get_somajo_doc(cand)
    return cand


def get_somajo_doc(cand: DataPoint) -> Dict[str, Any]:
    """
    Performs tokenization and sentence splitting using SoMaJo on the text of the DataPoint
    :param cand: DataPoint with at least a text field
    :return: Dictionary containing SoMaJo output, token list and sentences
    """
    load_somajo_model()
    somajo_doc: List[List[Token]] = list(nlp_somajo.tokenize_text([cand.text]))

    doc = {
        'doc': somajo_doc,
        'tokens': get_somajo_doc_tokens(somajo_doc),
        'sentences': get_somajo_doc_sentences(somajo_doc, cand.text)
    }
    return doc


def get_somajo_separate_sentence(cand: DataPoint) -> bool:
    """
    Checks based on the SoMaJo sentence splitting, whether the trigger and the argument of the DataPoint are in the
    same sentence.
    :param cand: DataPoint
    :return: Whether trigger and argument are in the same sentence according to SoMaJo
    """
    assert 'somajo_doc' in cand, 'You need to run get_somajo_doc first and add somajo_doc to the dataframe.'
    if len(cand.somajo_doc['sentences']) == 1:
        return False
    same_sentence: bool = False
    tolerance: int = 0
    trigger: Dict[str, Any] = cand.trigger
    argument: Dict[str, Any] = cand.argument
    somajo_doc: Dict[str, Any] = cand.somajo_doc
    for sentence in somajo_doc['sentences']:
        sentence_start = sentence['char_start']
        sentence_end = sentence['char_end']
        m_start: int = min(trigger['char_start'], argument['char_start'])
        m_end: int = max(trigger['char_end'], argument['char_end'])

        if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance:
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
    :return: Entities that are in the same sentence as the trigger-argument pair.
    """
    assert 'somajo_doc' in cand, "Need somajo_doc to retrieve sentence entities"
    tolerance: int = 0
    trigger: Dict[str, Any] = cand.trigger
    argument: Dict[str, Any] = cand.argument
    somajo_doc: Dict[str, Any] = cand.somajo_doc
    entities: List[Dict[str, Any]] = cand.entities
    for sentence in somajo_doc['sentences']:
        sentence_start = sentence['char_start']
        sentence_end = sentence['char_end']
        m_start: int = min(trigger['char_start'], argument['char_start'])
        m_end: int = max(trigger['char_end'], argument['char_end'])

        if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance:
            return [entity for entity in entities
                    if sentence_start <= entity['char_start'] + tolerance and
                    entity['char_end'] <= sentence_end + tolerance]
    return []


def is_multiple_same_event_type(cand: DataPoint) -> bool:
    between_text: str = get_between_text(cand)
    trigger: Dict[str, Any] = cand.trigger
    trigger_text: str = trigger['text']
    if trigger_text in between_text:
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
