from typing import Dict, List

from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


def get_entity(entity_id, entities):
    entity = next((x for x in entities if x['id'] == entity_id), None)
    if entity is None:
        raise Exception(f'The entity_id {entity_id} was not found in:\n {entities}')
    else:
        return entity


@preprocessor()
def get_trigger(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    cand.trigger = trigger
    return cand


@preprocessor()
def get_argument(cand: DataPoint) -> DataPoint:
    argument = get_entity(cand.argument_id, cand.entities)
    cand.argument = argument
    return cand


@preprocessor()
def get_left_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    cand.trigger_left_tokens = get_windowed_left_tokens(trigger, cand.tokens)

    # Only relevant for event argument role classification
    if 'argument_id' in cand:
        argument = get_entity(cand.argument_id, cand.entities)
        cand.argument_left_tokens = get_windowed_left_tokens(argument, cand.tokens)
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
    cand.trigger_left_tokens = get_windowed_right_tokens(trigger, cand.tokens)

    # Only relevant for event argument role classification
    if 'argument_id' in cand:
        argument = get_entity(cand.argument_id, cand.entities)
        cand.argument_left_tokens = get_windowed_right_tokens(argument, cand.tokens)
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
        cand.between_tokens = []
        return cand

    cand.between_tokens = cand.tokens[start:end]
    return cand


@preprocessor()
def get_between_distance(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    argument = get_entity(cand.argument_id, cand.entities)
    cand.between_distance = get_entity_distance(trigger, argument)
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
    cand.entity_type_freqs = entity_type_freqs
    return cand


def get_windowed_entity_type_freqs(entities, entity=None, window_size: int = None) -> Dict[str, int]:
    entity_type_freqs: Dict[str, int] = {}

    # Token based start, end numbers
    if entity is not None and window_size is not None:
        window_start = max(entity['start']-window_size, 0)
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
