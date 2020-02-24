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
    end = trigger['start']
    cand.trigger_left_tokens = cand.tokens[0:end]

    # Only relevant for event argument role classification
    if 'argument_id' in cand:
        argument = get_entity(cand.argument_id, cand.entities)
        end = argument['start']
        cand.argument_left_tokens = cand.tokens[0:end]
    return cand


@preprocessor()
def get_right_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    start = trigger['end']
    cand.trigger_left_tokens = cand.tokens[start:]

    # Only relevant for event argument role classification
    if 'argument_id' in cand:
        argument = get_entity(cand.argument_id, cand.entities)
        start = argument['end']
        cand.argument_left_tokens = cand.tokens[start:]
    return cand


@preprocessor()
def get_between_tokens(cand: DataPoint) -> DataPoint:
    trigger = get_entity(cand.trigger_id, cand.entities)
    argument = get_entity(cand.argument_id, cand.entities)

    # TODO: what if they overlap
    if trigger['end'] <= argument['start']:
        start = trigger['end']
        end = argument['start']
    else:
        start = argument['end']
        end = trigger['start']

    cand.between_tokens = cand.tokens[start:end]
    return cand


def get_entity_distance(entities, entity1_id, entity2_id) -> int:
    entity1 = get_entity(entity1_id, entities)
    entity1_start = entity1['start']
    entity1_end = entity1['end']

    entity2 = get_entity(entity2_id, entities)
    entity2_start = entity2['start']
    entity2_end = entity2['end']
    # TODO can entities overlap?
    if entity1_end <= entity2_start:
        return entity2_start - entity1_end
    elif entity2_end <= entity1_start:
        return entity1_start - entity2_end
    else:
        print(f"Overlapping entities {entity1} and {entity2}")
        return 0
