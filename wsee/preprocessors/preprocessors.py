from typing import Optional

from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


def get_index_for_id(obj_id, obj_list):
    for idx, obj in enumerate(obj_list):
        if obj['id'] == obj_id:
            return idx
    raise Exception(f'obj_id was not found in obj_list. The value of obj_id was: {obj_id}.\nThe obj_list: {obj_list}.')


def get_entity(row, entity_id):
    # future preprocessor function
    entity_idx = get_index_for_id(entity_id, row.entities)
    return row.entities[entity_idx]


def get_entity_text(row, entity_id):
    return get_entity(row, entity_id)['text']


def get_entity_type(row, entity_id):
    return get_entity(row, entity_id)['entity_type']


def get_entity_left_tokens(row, entity_id, window_size: int = None):
    if window_size is None:
        window_start = 0
    else:
        window_start = max(0, entity_start - window_size)
    window_end = get_entity(row, entity_id)['start']
    return row.tokens[window_start:window_end]


def get_entity_right_tokens(row, entity_id, window_size: int = None):
    window_start = get_entity(row, entity_id)['end']
    if window_size is None:
        window_end = len(row.tokens)
    else:
        window_end = min(len(row.tokens), window_start + window_size)
    return row.tokens[window_start:window_end]


def get_entity_distance(row, entity1_id, entity2_id):
    entity1 = get_entity(row, entity1_id)
    entity1_start = entity1['start']
    entity1_end = entity1['end']

    entity2 = get_entity(row, entity2_id)
    entity2_start = entity2['start']
    entity2_end = entity2['end']
    # TODO can entities overlap?
    if entity1_end <= entity2_start:
        return entity2_start - entity1_end
    elif entity2_end <= entity1_start:
        return entity1_start - entity2_end
    else:
        print(f"WHAT THE HELL: overlapping entities {entity1} and {entity2}")
        return 0


def get_trigger_arg_pair(row, trigger_id, argument_id):
    trigger = get_entity(row, trigger_id)
    argument = get_entity(row, argument_id)
    return trigger, argument
