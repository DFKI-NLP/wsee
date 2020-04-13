import re
import stanfordnlp
import spacy
import logging
import numpy as np

from typing import Dict, List, Optional, Callable
from somajo import SoMaJo
from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint
from wsee.preprocessors.pattern_event_processor import escape_regex_chars

punctuation_marks = ["<", "(", "[", "{", "\\", "^", "-", "=", "$", "!", "|",
                     "]", "}", ")", "?", "*", "+", ".", ",", ":", ";", ">",
                     "_", "#", "/"]

nlp_stanford: Optional[stanfordnlp.Pipeline] = None
nlp_spacy: Optional[Callable] = None
nlp_somajo: Optional[SoMaJo] = None


def load_stanford_model():
    global nlp_stanford
    if nlp_stanford is None:
        nlp_stanford = stanfordnlp.Pipeline(lang='de')


def load_spacy_model():
    global nlp_spacy
    if nlp_spacy is None:
        nlp_spacy = spacy.load('de_core_news_md')


def load_somajo_model():
    global nlp_somajo
    if nlp_somajo is None:
        nlp_somajo = SoMaJo("de_CMC", split_camel_case=True)


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


def get_entity_text_and_type(entity_id, entities):
    entity = get_entity(entity_id, entities)
    return entity['text'], entity['entity_type']


@preprocessor()
def get_trigger_text(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    cand['trigger_text'] = trigger['text']
    return cand


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
def get_argument_text(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)
    cand['argument_text'] = argument['text']
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
def get_trigger_left_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    cand['trigger_left_tokens'] = get_windowed_left_tokens(trigger, cand.tokens)
    return cand


@preprocessor()
def get_trigger_left_pos(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    if 'pos' in cand:
        cand['trigger_left_pos'] = get_windowed_left_pos(trigger, cand.pos)
    return cand


@preprocessor()
def get_argument_left_tokens(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)
    cand['argument_left_tokens'] = get_windowed_left_tokens(argument, cand.tokens)
    return cand


@preprocessor()
def get_argument_left_pos(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)
    if 'pos' in cand:
        cand['argument_left_pos'] = get_windowed_left_pos(argument, cand.pos)
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


@preprocessor()
def get_trigger_right_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    cand['trigger_right_tokens'] = get_windowed_right_tokens(trigger, cand.tokens)
    return cand


@preprocessor()
def get_trigger_right_pos(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    if 'pos' in cand:
        cand['trigger_right_pos'] = get_windowed_right_pos(trigger, cand.pos)
    return cand


@preprocessor()
def get_argument_right_tokens(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)
    cand['argument_right_tokens'] = get_windowed_right_tokens(argument, cand.tokens)
    return cand


@preprocessor()
def get_argument_right_pos(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)
    if 'pos' in cand:
        cand['argument_right_pos'] = get_windowed_right_pos(argument, cand.pos)
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
        logging.debug(f"Trigger {trigger['text']}({trigger['start']}, {trigger['end']}) and "
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


@preprocessor()
def get_all_trigger_distances(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)
    all_trigger_distances = {}
    for event_trigger in cand.event_triggers:
        trigger_id = event_trigger['id']
        if trigger_id != cand.argument_id:
            trigger = get_entity(trigger_id, cand.entities)
            distance = get_entity_distance(trigger, argument)
            all_trigger_distances[trigger_id] = distance
    cand['all_trigger_distances'] = all_trigger_distances
    return cand


@preprocessor()
def get_sentence_trigger_distances(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)

    load_somajo_model()
    somajo_doc = list(nlp_somajo.tokenize_text([cand.text]))

    sentences = get_somajo_doc_sentences(somajo_doc)

    sentence_trigger_distances = {}
    for event_trigger in cand.event_triggers:
        trigger_id = event_trigger['id']
        if trigger_id != cand.argument_id:
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
                m_start = min(trigger['char_start'], argument['char_start'])
                m_end = max(trigger['char_end'], argument['char_end'])
                if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance:
                    distance = get_entity_distance(trigger, argument)
                    sentence_trigger_distances[trigger_id] = distance
                    break
    cand['sentence_trigger_distances'] = sentence_trigger_distances
    return cand


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
    cand['mixed_ner'] = mixed_ner
    cand['mixed_ner_spans'] = mixed_ner_spans
    return cand


def get_spacy_doc_tokens(doc):
    return [token.text for token in doc]


def get_stanford_doc_tokens(doc):
    return [token.text for sentence in doc.sentences for token in sentence.tokens]


def get_somajo_doc_tokens(doc):
    return [token.text for sentence in doc for token in sentence]


def get_spacy_doc_sentences(doc):
    return [s.text for s in doc.sents]


def get_stanford_doc_sentences(doc):
    # introduces whitespaces
    # see: https://github.com/stanfordnlp/stanfordnlp/blob/dev/stanfordnlp/models/common/doc.py
    # to get original sentence text
    return [" ".join([token.text for token in sentence.tokens]) for sentence in doc.sentences]


def get_somajo_doc_sentences(doc):
    # introduces whitespaces
    return [" ".join([token.text for token in sentence]) for sentence in doc]


@preprocessor()
def get_spacy_doc(cand: DataPoint) -> DataPoint:
    load_spacy_model()
    spacy_doc = nlp_spacy(cand.text)
    doc = {
        'doc': spacy_doc,
        'tokens': get_spacy_doc_tokens(spacy_doc),
        'sentences': get_spacy_doc_sentences(spacy_doc),  # preserves sentence texts
        'trigger_text': nlp_spacy(get_entity(cand.trigger_id, cand.entities)['text']),
    }
    if 'argument_id' in cand:
        doc['argument_text'] = get_entity(cand.argument_id, cand.entities)['text']
    cand['spacy_doc'] = doc
    return cand


@preprocessor()
def get_stanford_doc(cand: DataPoint) -> DataPoint:
    load_stanford_model()
    stanford_doc = nlp_stanford(cand.text)
    trigger_text = get_entity(cand.trigger_id, cand.entities)['text']
    tokenized_trigger = get_stanford_doc_tokens(nlp_stanford(trigger_text))
    doc = {
        'doc': stanford_doc,
        'tokens': get_stanford_doc_tokens(stanford_doc),
        'sentences': get_stanford_doc_sentences(stanford_doc),
        'trigger': " ".join(tokenized_trigger),
    }
    if 'argument_id' in cand:
        argument_text = get_entity(cand.argument_id, cand.entities)['text']
        tokenized_argument = get_stanford_doc_tokens(nlp_stanford(argument_text))
        doc['argument'] = " ".join(tokenized_argument)
    cand['stanford_doc'] = doc
    return cand


@preprocessor()
def get_somajo_doc(cand: DataPoint) -> DataPoint:
    load_somajo_model()
    somajo_doc = list(nlp_somajo.tokenize_text([cand.text]))
    trigger_text = get_entity(cand.trigger_id, cand.entities)['text']
    tokenized_trigger = get_somajo_doc_tokens(nlp_somajo.tokenize_text([trigger_text]))
    doc = {
        'doc': somajo_doc,
        'tokens': get_somajo_doc_tokens(somajo_doc),
        'sentences': get_somajo_doc_sentences(somajo_doc),
        'trigger': " ".join(tokenized_trigger),
    }
    if 'argument_id' in cand:
        argument_text = get_entity(cand.argument_id, cand.entities)['text']
        tokenized_argument = get_somajo_doc_tokens(nlp_somajo.tokenize_text([argument_text]))
        doc['argument'] = " ".join(tokenized_argument)
    cand['somajo_doc'] = doc
    return cand


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
