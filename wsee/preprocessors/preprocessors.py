import re
import stanfordnlp
import spacy
from typing import Dict, List

from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint
from wsee.preprocessors.pattern_event_processor import escape_regex_chars

punctuation_marks = ["<", "(", "[", "{", "\\", "^", "-", "=", "$", "!", "|",
                     "]", "}", ")", "?", "*", "+", ".", ",", ":", ";", ">",
                     "_", "#", "/"]


def get_entity_idx(entity_id, entities):
    entity_idx = next((idx for idx, x in enumerate(entities) if x['id'] == entity_id), None)
    if entity_idx is None:
        raise Exception(f'The entity_id {entity_id} was not found in:\n {entities}')
    else:
        return entity_idx


def get_entity(entity_id, entities):
    entity_idx = get_entity_idx(entity_id, entities)
    entity = entities[entity_idx]
    return entity


@preprocessor()
def get_trigger(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    cand['trigger'] = trigger
    return cand


@preprocessor()
def get_trigger_idx(cand: DataPoint) -> DataPoint:
    trigger_idx = get_entity_idx(cand.trigger_id, cand.entities)
    cand['trigger_idx'] = trigger_idx
    return cand


@preprocessor()
def get_argument(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)
    cand['argument'] = argument
    return cand


@preprocessor()
def get_argument_idx(cand: DataPoint) -> DataPoint:
    argument_idx = get_entity_idx(cand.argument_id, cand.entities)
    cand['argument_idx'] = argument_idx
    return cand


@preprocessor()
def get_left_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    cand['trigger_left_tokens'] = get_windowed_left_tokens(trigger, cand.tokens)

    # Only relevant for event argument role classification
    if 'argument_id' in cand:
        argument = get_entity(cand.argument_id, cand.entities)
        cand['argument_left_tokens'] = get_windowed_left_tokens(argument, cand.tokens)
    return cand


def get_windowed_left_tokens(entity, tokens, window_size: int = None) -> List[str]:
    window_end = entity['start']
    if window_size is None:
        window_start = 0
    else:
        window_start = max(0, window_end - window_size)

    return tokens[window_start:window_end]


@preprocessor()
def get_right_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    cand['trigger_right_tokens'] = get_windowed_right_tokens(trigger, cand.tokens)

    # Only relevant for event argument role classification
    if 'argument_id' in cand:
        argument = get_entity(cand.argument_id, cand.entities)
        cand['argument_right_tokens'] = get_windowed_right_tokens(argument, cand.tokens)
    return cand


def get_windowed_right_tokens(entity, tokens, window_size: int = None) -> List[str]:
    window_start = entity['end']
    if window_size is None:
        window_end = len(tokens)
    else:
        window_end = min(len(tokens), window_start + window_size)

    return tokens[window_start:window_end]


@preprocessor()
def get_between_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    argument = get_entity(cand.argument_id, cand.entities)

    if trigger['end'] <= argument['start']:
        start = trigger['end']
        end = argument['start']
    elif argument['end'] <= trigger['start']:
        start = argument['end']
        end = trigger['start']
    else:
        print(f"Trigger {trigger['text']}({trigger['start']}, {trigger['end']}) and "
              f"argument {argument['text']}({argument['start']}, {argument['end']}) are overlapping.")
        cand['between_tokens'] = []
        return cand

    cand['between_tokens'] = cand.tokens[start:end]
    return cand


@preprocessor()
def get_between_distance(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    argument = get_entity(cand.argument_id, cand.entities)
    cand['between_distance'] = get_entity_distance(trigger, argument)
    return cand


def get_entity_distance(entity1, entity2) -> int:
    if entity1['end'] <= entity2['start']:
        return entity2['start'] - entity1['end']
    elif entity2['end'] <= entity1['start']:
        return entity1['start'] - entity2['end']
    else:
        print(f"Overlapping entities {entity1['text']}({entity1['start']}, {entity1['end']}) and "
              f"{entity2['text']}({entity2['start']}, {entity2['end']})")
        return 0


@preprocessor()
def get_entity_type_freqs(cand: DataPoint) -> DataPoint:
    entity_type_freqs = get_windowed_entity_type_freqs(entities=cand.entities)
    cand['entity_type_freqs'] = entity_type_freqs
    return cand


def get_windowed_entity_type_freqs(entities, entity=None, window_size: int = None) -> Dict[str, int]:
    entity_type_freqs: Dict[str, int] = {}

    # Token based start, end numbers
    if entity is not None and window_size is not None:
        window_start = max(entity['start'] - window_size, 0)
        window_end = min(entity['end'] + window_size, len(entities))
    else:
        window_start = 0
        window_end = len(entities)

    # Reduce entities list to entities within token based window
    relevant_entity_types = [entity['entity_type'] for entity in entities
                             if entity['start'] >= window_start and entity['end'] <= window_end]

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
            assert cand.text[match_start:match_end] == entity['text']
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
    cand['mixed_ner'] = mixed_ner
    cand['mixed_ner_spans'] = mixed_ner_spans
    return cand


def get_stanford_model():
    return stanfordnlp.Pipeline(lang='de')


def get_spacy_model():
    return spacy.load('de_core_news_md')


@preprocessor()
def get_stanford_doc(cand: DataPoint) -> DataPoint:
    nlp_stanford = get_stanford_model()
    cand['stanford_doc'] = nlp_stanford(cand.text)
    return cand


@preprocessor()
def get_spacy_doc(cand: DataPoint) -> DataPoint:
    nlp_spacy = get_spacy_model()
    cand['spacy_doc'] = nlp_spacy(cand.text)
    return cand
