from snorkel.labeling import labeling_function
from wsee.preprocessors.preprocessors import *
from wsee.preprocessors.pattern_event_processor import parse_pattern_file, find_best_pattern_match, location_subtypes
from wsee.labeling import event_trigger_lfs
from wsee.utils import utils

location = 0
delay = 1
direction = 2
start_loc = 3
end_loc = 4
start_date = 5
end_date = 6
cause = 7
jam_length = 8
route = 9
no_arg = 10
ABSTAIN = -1

labels = {
    'location': 0,
    'delay': 1,
    'direction': 2,
    'start_loc': 3,
    'end_loc': 4,
    'start_date': 5,
    'end_date': 6,
    'cause': 7,
    'jam_length': 8,
    'route': 9,
    'no_arg': 10
}

event_type_lf_map: Dict[int, Any] = {
    event_trigger_lfs.Accident: event_trigger_lfs.lf_accident_context,
    event_trigger_lfs.CanceledRoute: event_trigger_lfs.lf_canceledroute_cat,
    event_trigger_lfs.CanceledStop: event_trigger_lfs.lf_canceledstop_cat,
    event_trigger_lfs.Delay: event_trigger_lfs.lf_delay_cat,
    event_trigger_lfs.Obstruction: event_trigger_lfs.lf_obstruction_cat,
    event_trigger_lfs.RailReplacementService: event_trigger_lfs.lf_railreplacementservice_cat,
    event_trigger_lfs.TrafficJam: event_trigger_lfs.lf_trafficjam_cat
}

event_type_location_type_map: Dict[int, List[str]] = {
    event_trigger_lfs.Accident: ['location', 'location_street', 'location_city', 'location_route'],
    event_trigger_lfs.CanceledRoute: ['location_route'],
    event_trigger_lfs.CanceledStop: ['location_stop'],
    event_trigger_lfs.Delay: ['location_route'],
    event_trigger_lfs.Obstruction: ['location', 'location_street', 'location_city', 'location_route'],
    event_trigger_lfs.RailReplacementService: ['location_route'],
    event_trigger_lfs.TrafficJam: ['location', 'location_street', 'location_city', 'location_route']
}

original_rules = parse_pattern_file('/Users/phuc/develop/python/wsee/data/event-patterns-annotated-britta-olli.txt')
general_location_rules = parse_pattern_file('/Users/phuc/develop/python/wsee/data/event-patterns-phuc.txt')
traffic_event_causes = utils.parse_gaz_file('/Users/phuc/develop/python/wsee/data/traffic_event_causes.gaz')


# utility function
def check_required_args(entity_freqs):
    if any(loc_type in entity_freqs
           for loc_type in ['location', 'location_route', 'location_street', 'location_stop', 'location_city']) and \
            'trigger' in entity_freqs:
        return True
    else:
        return False


def is_nearest_trigger(between_distance: int, all_trigger_distances: Dict[str, int]):
    if all_trigger_distances:
        min_distance = min(all_trigger_distances.values())
        if between_distance <= min_distance:
            return True
    return False


# location role
# TODO: check for event types
#  Accident (loc, loc_city, loc_street), CanceledRoute (loc_route), CanceledStop (loc_stop), Delay (loc_route),
#  Obstruction (all except loc_stop), RailReplacementService (loc_route),
#  TrafficJam (loc, loc_city, loc_street, loc_route)

def lf_location(x, same_sentence=True, nearest=False, check_event_type=True,
                entity_types=('location', 'location_street', 'location_route', 'location_city', 'location_stop')):
    """
    Generalized labeling function for location argument type.
    :param x: DataPoint containing additional columns defined via the preprocessors argument.
    :param same_sentence: Abstain if trigger and argument are not in the same sentence.
    :param nearest: Abstain if trigger is not the nearest trigger of the argument.
    :param check_event_type: Check if argument entity type matches trigger event type.
    :param entity_types: General list of allowed entity types for the argument.
    :return: location label or ABSTAIN
    """
    between_distance = get_between_distance(x)
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    all_trigger_distances = get_all_trigger_distances(x)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            (x.trigger['text'] == 'aus' and x.trigger['start'] < x.argument['start']):
        return ABSTAIN
    if same_sentence:
        if lf_somajo_separate_sentence(x) != ABSTAIN:
            return ABSTAIN
    if nearest:
        if same_sentence:
            if not is_nearest_trigger(between_distance, sentence_trigger_distances):
                return ABSTAIN
        else:
            if not is_nearest_trigger(between_distance, all_trigger_distances):
                return ABSTAIN

    arg_entity_type = x.argument['entity_type']
    if arg_entity_type not in entity_types:
        return ABSTAIN

    if not check_event_type:
        return location
    else:
        for event_class, location_types in event_type_location_type_map.items():
            if arg_entity_type in location_types and event_type_lf_map[event_class](x) == event_class:
                if event_class == event_trigger_lfs.CanceledStop and between_distance > 10:
                    return ABSTAIN
                else:
                    return location
    return ABSTAIN


@labeling_function(pre=[])
def lf_location_chained(x):
    if any(location_lf(x) == location for location_lf in
           [
               lf_location_adjacent_markers,
               lf_location_beginning_street_stop_route,
               lf_location_first_sentence,
               lf_location_first_sentence_nearest,
               lf_location_first_sentence_street_stop_route,
               lf_location_first_sentence_priorities,
               lf_event_patterns,
               lf_event_patterns_general_location
           ]):
        return location
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_location_adjacent_markers(x):
    between_distance = get_between_distance(x)
    between_tokens = get_between_tokens(x)
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and \
            lf_direction_type(x) == ABSTAIN and no_entity_in_between(x):
        if (':' in between_tokens and between_distance <= 1) or \
                (x.trigger['start'] < x.argument['start'] and between_distance < 3 and
                 no_entity_in_between(x) and
                 (any(token for token in between_tokens if token in ['auf', 'bei', 'in']))):
            return lf_location(x)
    return ABSTAIN


@labeling_function(pre=[])
def lf_location_beginning_street_stop_route(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and \
            lf_direction_type(x) == ABSTAIN and x.argument['start'] == 0 and \
            x.argument['entity_type'] not in ['location_city', 'location']:
        return lf_location(x)
    return ABSTAIN


@labeling_function(pre=[])
def lf_location_first_sentence(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and \
            lf_direction_type(x) == ABSTAIN:
        first_location_entity = get_first_of_entity_types(
            get_sentence_entities(x),
            ['location', 'location_route', 'location_street', 'location_stop', 'location_city'])
        if first_location_entity and first_location_entity['id'] == x.argument['id']:
            return lf_location(x)
    return ABSTAIN


@labeling_function(pre=[])
def lf_location_first_sentence_nearest(x):
    between_distance = get_between_distance(x)
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and \
            lf_direction_type(x) == ABSTAIN:
        first_location_entity = get_first_of_entity_types(
            get_sentence_entities(x),
            ['location', 'location_route', 'location_street', 'location_stop', 'location_city'])
        if first_location_entity and first_location_entity['id'] == x.argument['id'] and \
                is_nearest_trigger(between_distance, sentence_trigger_distances):
            return lf_location(x)
    return ABSTAIN


@labeling_function(pre=[])
def lf_location_first_sentence_not_city(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and \
            lf_direction_type(x) == ABSTAIN:
        first_location_entity = get_first_of_entity_types(
            get_sentence_entities(x), ['location', 'location_route', 'location_street', 'location_stop'])
        if first_location_entity and first_location_entity['id'] == x.argument['id']:
            return lf_location(x)
    return ABSTAIN


@labeling_function(pre=[])
def lf_location_first_sentence_street_stop_route(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and \
            lf_direction_type(x) == ABSTAIN:
        first_location_entity = get_first_of_entity_types(
            get_sentence_entities(x), ['location_route', 'location_street', 'location_stop'])
        if first_location_entity and first_location_entity['id'] == x.argument['id']:
            return lf_location(x)
    return ABSTAIN


@labeling_function(pre=[])
def lf_location_first_sentence_priorities(x):
    """
    Looks at the first location entity. If it is a (general) location/ city entity and a
    street/stop/route location entity follows immediately, the location argument of interest is
    probably the latter. There choose that second location entity as the argument.
    :param x:
    :return:
    """
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and \
            lf_direction_type(x) == ABSTAIN:
        first_location_entity = get_first_of_entity_types(
            get_sentence_entities(x),
            ['location', 'location_route', 'location_street', 'location_stop', 'location_city'])
        first_street_stop_route = get_first_of_entity_types(
            get_sentence_entities(x), ['location_route', 'location_stop', 'location_street'])
        if first_location_entity:
            if first_street_stop_route:
                first_distance = get_entity_distance(first_location_entity, first_street_stop_route)
                if first_location_entity['id'] != first_street_stop_route['id'] and first_distance < 2:
                    if first_street_stop_route['id'] == x.argument['id']:
                        return lf_location(x)
                    else:
                        return ABSTAIN
                elif first_location_entity['id'] == first_street_stop_route['id'] and \
                        first_street_stop_route['id'] == x.argument['id']:
                    return lf_location(x)
            elif first_location_entity['id'] == x.argument['id']:
                return lf_location(x)
    return ABSTAIN


def get_first_of_entity_types(entities, entity_types: List[str]):
    first_entity = None
    for entity in entities:
        if entity['entity_type'] in entity_types and \
                (first_entity is None or first_entity['start'] > entity['start']):
            first_entity = entity
    return first_entity


def no_entity_in_between(x):
    no_in_between = True
    for entity in x.entities:
        if x.argument['start'] < entity['start'] < x.trigger['start'] or \
                x.trigger['start'] < entity['start'] < x.argument['start']:
            no_in_between = False
    return no_in_between


# delay role
@labeling_function(pre=[])
def lf_delay_event_sentence(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            'früher' in argument_right_tokens[:2] or \
            (lf_somajo_separate_sentence(x) != ABSTAIN and 'Zeitverlust' not in argument_left_tokens[-4:]):
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['duration']:
            if event_trigger_lfs.lf_delay_cat(x) == event_trigger_lfs.Delay or \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam or \
                    event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction or \
                    event_trigger_lfs.lf_accident_context(x) == event_trigger_lfs.Accident:
                return delay
    return ABSTAIN


# delay role
@labeling_function(pre=[])
def lf_delay_event_sentence_check(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            'früher' in argument_right_tokens[:2] or \
            (lf_somajo_separate_sentence(x) != ABSTAIN and 'Zeitverlust' not in argument_left_tokens[-4:]):
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['duration']:
            if event_trigger_lfs.lf_delay_cat(x) == event_trigger_lfs.Delay or \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam or \
                    event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction or \
                    event_trigger_lfs.lf_accident_context(x) == event_trigger_lfs.Accident:
                if get_entity_distance(x.trigger, x.argument) > 0 and \
                        x.argument['text'].islower() and \
                        x.argument['start'] < x.trigger['start']:
                    # Rationale here: duration can only occur after a trigger verb,
                    # e.g. "verspätet sich um 15 min"
                    # Only case, where a lower case trigger can occur after the duration "15 später", they are
                    # adjacent
                    return ABSTAIN
                else:
                    return delay
    return ABSTAIN


# direction role
# TODO: no location_route, location_street and check for event types
#  Accident (loc, loc_city), CanceledRoute (loc_stop, loc_city), CanceledStop (none), Delay (loc_stop),
#  Obstruction (loc, loc_city), RailReplacementService (loc_stop), TrafficJam (loc, loc_city)
@labeling_function(pre=[])
def lf_direction_type(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location', 'location_city', 'location_stop', 'location_street']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
                if any(token.lower() in ['nach', 'richtung', 'fahrtrichtung', '->']
                       for token in argument_left_tokens[-1:]) or \
                        x.argument['text'].lower() in ['richtung', 'richtungen', 'stadteinwärts', 'stadtauswärts',
                                                       'beide richtungen', 'gegenrichtung', 'je richtung']:
                    return direction
                elif len(argument_left_tokens) > 2 and argument_left_tokens[-1] == '-':
                    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
                    if argument_left_ner[-3][2:] == 'LOCATION_STREET' and \
                            argument_left_ner[-2][2:] in ['LOCATION', 'LOCATION_CITY']:
                        return direction
    return ABSTAIN


@labeling_function(pre=[])
def lf_direction_order(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
        return ABSTAIN
    if x.argument['start'] > x.trigger['start']:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location', 'location_city', 'location_stop', 'location_street']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
                if any(token.lower() in ['nach', 'richtung', 'fahrtrichtung', '->']
                       for token in argument_left_tokens[-1:]) or \
                        x.argument['text'].lower() in ['richtung', 'richtungen', 'stadteinwärts', 'stadtauswärts',
                                                       'beide richtungen', 'beiden richtungen', 'gegenrichtung',
                                                       'je richtung']:
                    return direction
                elif len(argument_left_tokens) > 2 and argument_left_tokens[-1] == '-':
                    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
                    if argument_left_ner[-3][2:] == 'LOCATION_STREET' and \
                            argument_left_ner[-2][2:] in ['LOCATION', 'LOCATION_CITY']:
                        return direction
    return ABSTAIN


@labeling_function(pre=[])
def lf_loc_stop_direction_order(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
        return ABSTAIN
    if x.argument['start'] > x.trigger['start']:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location_stop']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                    (any(token.lower() in ['nach', 'richtung', 'fahrtrichtung', '->']
                         for token in argument_left_tokens[-1:]) or
                     (argument_left_ner and argument_left_ner[-1][2:] == 'LOCATION_ROUTE')) and \
                    (event_trigger_lfs.lf_canceledroute_cat(x) == event_trigger_lfs.CanceledRoute or
                     event_trigger_lfs.lf_delay_cat(x) == event_trigger_lfs.Delay or
                     event_trigger_lfs.lf_railreplacementservice_cat(x) == event_trigger_lfs.RailReplacementService):
                return direction
    return ABSTAIN


@labeling_function(pre=[])
def lf_loc_loc_city_direction_type(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            lf_somajo_separate_sentence(x) != ABSTAIN:
        return ABSTAIN
    if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location', 'location_city', 'location_stop']:
            if (any(token.lower() in ['nach', 'richtung', 'fahrtrichtung', '->']
                    for token in argument_left_tokens[-1:]) or
                x.argument['text'].lower() in ['richtung', 'richtungen', 'stadteinwärts', 'stadtauswärts',
                                               'beide richtungen', 'beiden richtungen', 'gegenrichtung',
                                               'je richtung'] or
                (argument_left_ner and argument_left_ner[-1][2:] == 'LOCATION_ROUTE')) and \
                    (event_trigger_lfs.lf_accident_context(x) == event_trigger_lfs.Accident or
                     event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction or
                     event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam):
                return direction
            elif len(argument_left_tokens) > 2 and argument_left_tokens[-1] == '-':
                argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
                if argument_left_ner[-3][2:] == 'LOCATION_STREET' and \
                        argument_left_ner[-2][2:] in ['LOCATION', 'LOCATION_CITY']:
                    return direction
    return ABSTAIN


@labeling_function(pre=[])
def lf_loc_loc_city_direction_order(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            lf_somajo_separate_sentence(x) != ABSTAIN:
        return ABSTAIN
    if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
        return ABSTAIN
    if x.argument['start'] > x.trigger['start']:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location', 'location_city', 'location_stop']:
            if (any(token.lower() in ['nach', 'richtung', 'fahrtrichtung', '->']
                    for token in argument_left_tokens[-1:]) or
                x.argument['text'].lower() in ['richtung', 'richtungen', 'stadteinwärts', 'stadtauswärts',
                                               'beide richtungen', 'beiden richtungen', 'gegenrichtung',
                                               'je richtung'] or
                (argument_left_ner and argument_left_ner[-1][2:] == 'LOCATION_ROUTE')) and \
                    (event_trigger_lfs.lf_accident_context(x) == event_trigger_lfs.Accident or
                     event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction or
                     event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam):
                return direction
            elif len(argument_left_tokens) > 2 and argument_left_tokens[-1] == '-':
                argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
                if argument_left_ner[-3][2:] == 'LOCATION_STREET' and \
                        argument_left_ner[-2][2:] in ['LOCATION', 'LOCATION_CITY']:
                    return direction
    return ABSTAIN


# start_loc
# TODO: no location_route and check for event types
#  Accident (loc, loc_city), CanceledRoute (loc_stop), CanceledStop (none), Delay (loc_stop),
#  Obstruction (all), RailReplacementService (loc_stop), TrafficJam (loc, loc_city, loc_street)
@labeling_function(pre=[])
def lf_start_location_type(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
    argument_right_ner = get_windowed_right_ner(x.argument, x.ner_tags)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            event_trigger_lfs.lf_canceledstop_cat(x) != ABSTAIN:
        return ABSTAIN
    if lf_somajo_separate_sentence(x) != ABSTAIN or lf_not_an_event(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        indicator_idx = next((idx for idx, token in enumerate(argument_left_tokens[-3:])
                              if token.lower() in ['zw', 'zw.', 'zwischen', 'ab', 'von']), -1)
        if (indicator_idx > -1 and
            all(left_ner[2:] not in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP', 'LOCATION_ROUTE']
                for left_ner in argument_left_ner[indicator_idx:])) or \
                (argument_right_tokens and argument_right_tokens[0] in ['-', '<', '>', '<>'] and
                 len(argument_right_ner) > 1 and
                 argument_right_ner[1][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP']):
            if argument_left_tokens and 'von' == argument_left_tokens[-1] and \
                    len(argument_left_ner) > 1 and \
                    argument_left_ner[-2][2:] in ['LOCATION_ROUTE']:
                return ABSTAIN
            else:
                return start_loc
    return ABSTAIN


@labeling_function(pre=[])
def lf_start_location_nearest(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
    argument_right_ner = get_windowed_right_ner(x.argument, x.ner_tags)
    between_distance = get_between_distance(x)
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            event_trigger_lfs.lf_canceledstop_cat(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        if is_nearest_trigger(between_distance, sentence_trigger_distances) and lf_not_an_event(x) == ABSTAIN:
            indicator_idx = next((idx for idx, token in enumerate(argument_left_tokens[-3:])
                                  if token.lower() in ['zw', 'zw.', 'zwischen', 'ab', 'von']), -1)
            if (indicator_idx > -1 and
                all(left_ner not in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP']
                    for left_ner in argument_left_ner[indicator_idx:][2:])) or \
                    (argument_right_tokens and argument_right_tokens[0] in ['-', '<', '>', '<>'] and
                     len(argument_right_ner) > 1 and
                     argument_right_ner[1][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP']):
                if argument_left_tokens and 'von' == argument_left_tokens[-1:] and \
                        len(argument_left_ner) > 1 and \
                        argument_left_ner[-2][2:] in ['LOCATION_ROUTE']:
                    return ABSTAIN
                else:
                    return start_loc
    return ABSTAIN


# end_loc
@labeling_function(pre=[])
def lf_end_location_type(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            event_trigger_lfs.lf_canceledstop_cat(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                ((any(token.lower() in ['zw.', 'zwischen'] for token in argument_left_tokens[-6:]) and
                  any(token.lower() in ['und', 'u.', '<', '>', '<>', '&'] for token in argument_left_tokens[-1:])) or
                 any(token.lower() in ['bis'] for token in argument_left_tokens[-1:]) or
                 (argument_left_tokens and '-' == argument_left_tokens[-1] and
                  len(argument_left_ner) > 1 and
                  argument_left_ner[-2][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP'])):
            return end_loc
    return ABSTAIN


@labeling_function(pre=[])
def lf_end_location_nearest(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
    between_distance = get_between_distance(x)
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        if is_nearest_trigger(between_distance, sentence_trigger_distances) and lf_not_an_event(x) == ABSTAIN and \
                ((any(token.lower() in ['zw.', 'zwischen'] for token in argument_left_tokens[-6:]) and
                  any(token.lower() in ['und', 'u.', '<', '>', '<>', '&'] for token in argument_left_tokens[-1:])) or
                 any(token.lower() in ['bis'] for token in argument_left_tokens[-1:]) or
                 (argument_left_tokens and '-' == argument_left_tokens[-1] and
                  len(argument_left_ner) > 1 and
                  argument_left_ner[-2][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP'])):
            return end_loc
    return ABSTAIN


# start_date
@labeling_function(pre=[])
def lf_start_date_type(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    argument_right_ner = get_windowed_right_ner(x.argument, x.ner_tags)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or 'Meldung' in argument_right_tokens \
            or lf_not_an_event(x) != ABSTAIN or lf_somajo_separate_sentence(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            if any(left_token == 'gültig' for left_token in argument_left_tokens[-4:]) and \
                    'ab' in argument_left_tokens[-3:]:
                return ABSTAIN
            if ((any(token.lower() in ['ab', 'von', 'vom'] for token in argument_left_tokens[-3:]) and
                 not any(token.lower() in ['bis'] for token in argument_left_tokens[-3:])) or
                    (argument_right_tokens and argument_right_tokens[0] in ['und', '/', '-', '->'] and
                     len(argument_right_ner) > 1 and argument_right_ner[1][2:] in ['DATE', 'TIME'])):
                return start_date
    return ABSTAIN


@labeling_function(pre=[])
def lf_start_date_nearest(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    between_distance = get_between_distance(x)
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    entity_trigger_distances = get_entity_trigger_distances(x)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or 'Meldung' in argument_right_tokens \
            or lf_not_an_event(x) != ABSTAIN or lf_end_date_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if any(left_token == 'gültig' for left_token in argument_left_tokens[-4:]) and \
                'ab' in argument_left_tokens[-3:]:
            return ABSTAIN
        if arg_entity_type in ['date', 'time']:
            if is_nearest_trigger(between_distance, sentence_trigger_distances) and \
                    entity_trigger_distances[arg_entity_type] and \
                    between_distance <= min(entity_trigger_distances[arg_entity_type]) and \
                    lf_not_an_event(x) == ABSTAIN and lf_end_date_type(x) == ABSTAIN:
                return start_date
    return ABSTAIN


@labeling_function(pre=[])
def lf_start_date_first(x):
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or 'Meldung' in argument_right_tokens \
            or lf_not_an_event(x) != ABSTAIN or \
            lf_somajo_separate_sentence(x) != ABSTAIN or lf_end_date_type(x) != ABSTAIN:
        return ABSTAIN
    first_date = get_first_of_entity_types(x.entities, ['date', 'time'])
    if check_required_args(x.entity_type_freqs) and first_date and first_date['id'] == x.argument['id']:
        return start_date
    else:
        return ABSTAIN


# end_date
@labeling_function(pre=[])
def lf_end_date_type(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or 'Meldung' in argument_right_tokens:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                    ((any(token.lower() in ['bis', 'endet', 'enden'] for token in argument_left_tokens[-3:]) and
                      not any(token.lower() in ['von', 'ab', 'vom'] for token in argument_left_tokens[-2:])) or
                     (argument_left_tokens and argument_left_tokens[-1] in ['und', '/', '-'] and
                      len(argument_left_ner) > 1 and argument_left_ner[1][2:] in ['DATE', 'TIME'])):
                return end_date
    return ABSTAIN


# cause
@labeling_function(pre=[])
def lf_cause_type(x):
    argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
    argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['trigger']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                    (event_trigger_lfs.check_cause_keywords(argument_left_tokens[-4:], x) or
                     (argument_right_tokens and argument_right_tokens[0].lower() in ['erzeugt', 'erzeugen'])):
                # TODO check trigger-arg order, some event types have higher priority
                #  often Accident cause for TrafficJam
                return cause
    return ABSTAIN


@labeling_function(resources=dict(cause_mapping=traffic_event_causes), pre=[])
def lf_cause_gaz_file(x, cause_mapping):
    # TODO add more cause pairs, coverage is 0%
    if x.argument['text'] in cause_mapping:
        if cause_mapping[x.argument['text']] == x.trigger['text']:
            return cause
    return ABSTAIN


# jam_length
@labeling_function(pre=[])
def lf_distance_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['distance']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam:
                return jam_length
    return ABSTAIN


@labeling_function(pre=[])
def lf_distance_nearest(x):
    between_distance = get_between_distance(x)
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    entity_trigger_distances = get_entity_trigger_distances(x)
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['distance']:
            if is_nearest_trigger(between_distance, sentence_trigger_distances) and \
                    entity_trigger_distances[arg_entity_type] and \
                    between_distance <= min(entity_trigger_distances[arg_entity_type]) and \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam:
                return jam_length
    return ABSTAIN


# route
@labeling_function(pre=[])
def lf_route_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location_route']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and \
                event_trigger_lfs.lf_canceledstop_cat(x) == event_trigger_lfs.CanceledStop:
            return route
    return ABSTAIN


@labeling_function(pre=[])
def lf_route_type_order(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location_route']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and \
                event_trigger_lfs.lf_canceledstop_cat(x) == event_trigger_lfs.CanceledStop and \
                x.argument['start'] < x.trigger['start']:
            return route
    return ABSTAIN


# no_args
@labeling_function(pre=[])
def lf_not_an_event(x):
    if event_trigger_lfs.lf_negative(x) == event_trigger_lfs.O:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[pre_spacy_doc])
def lf_spacy_separate_sentence(x):
    # proof of concept that makes use of spaCy sentence splitter parsing feature
    # see: https://github.com/explosion/spaCy/blob/master/examples/information_extraction/entity_relations.py
    same_sentence = False
    for sentence in x.spacy_doc['sentences']:
        # TODO use check_spans from preprocessors to make sure
        # edge case: two very similar sentences that both contain trigger text and arg text

        if x.spacy_doc['trigger'] in sentence and x.spacy_doc['argument'] in sentence:
            same_sentence = True
            break
    if same_sentence:
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[pre_stanford_doc])
def lf_stanford_separate_sentence(x):
    same_sentence = False

    for sentence in x.stanford_doc['sentences']:
        # edge case: two very similar sentences that both contain trigger text and arg text
        if x.stanford_doc['trigger'] in sentence and x.stanford_doc['argument'] in sentence:
            same_sentence = True
            break
    if same_sentence:
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[])
def lf_somajo_separate_sentence(x):
    same_sentence = False
    text = ""
    tolerance = 0
    for sentence, sent_tokens in zip(x.somajo_doc['sentences'], x.somajo_doc['doc']):
        sentence_start = len(text)
        text += sentence
        sentence_end = len(text)
        # allow for some tolerance for wrong whitespaces: number of punctuation marks, new lines  for now
        # factor 2 because for each punctuation marks we are adding at most 2 wrong whitespaces
        tolerance += 2 * len([token for token in sent_tokens if token.text in punctuation_marks]) + sentence.count('\n')
        m_start = min(x.trigger['char_start'], x.argument['char_start'])
        m_end = max(x.trigger['char_end'], x.argument['char_end'])

        somajo_trigger = x.somajo_doc['entities'][x.trigger['id']]
        somajo_argument = x.somajo_doc['entities'][x.argument['id']]

        if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance and \
                somajo_trigger in sentence and somajo_argument in sentence:
            same_sentence = True
            break
    if same_sentence:
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[])
def lf_not_nearest_event(x):
    between_distance = get_between_distance(x)
    all_trigger_distances = get_all_trigger_distances(x)
    if is_nearest_trigger(between_distance, all_trigger_distances):
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[])
def lf_not_nearest_same_sentence_event(x):
    between_distance = get_between_distance(x)
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    if is_nearest_trigger(between_distance, sentence_trigger_distances):
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[])
def lf_not_nearest_event_argument(x):
    between_distance = get_between_distance(x)
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    entity_trigger_distances = get_entity_trigger_distances(x)
    argument_type = x.argument['entity_type']
    if argument_type not in ['location', 'location_city', 'location_route', 'location_stop', 'location_street', 'date',
                             'time', 'duration', 'distance', 'trigger']:
        return no_arg
    if is_nearest_trigger(between_distance, sentence_trigger_distances) and \
            'entity_trigger_distances' in x and \
            entity_trigger_distances[argument_type] and \
            between_distance <= min(entity_trigger_distances[argument_type]):
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[])
def lf_overlapping(x):
    between_distance = get_between_distance(x)
    if between_distance < 0:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_too_far_40(x):
    between_distance = get_between_distance(x)
    if between_distance > 40:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_multiple_same_event_type(x):
    # TODO use keyword list and loop over them instead of trying to exactly match the trigger text
    between_tokens = get_between_tokens(x)
    trigger_text = x.trigger['text']
    if trigger_text in between_tokens:
        return no_arg
    else:
        return ABSTAIN


# general
@labeling_function(pre=[pre_spacy_doc])
def lf_dependency(x):
    # proof of concept that makes use of spaCy dependency parsing feature
    # see: https://github.com/explosion/spaCy/blob/master/examples/information_extraction/entity_relations.py
    for token in x.spacy_doc:
        # TODO match tokenization and fix spans
        if token.text in x.trigger['text']:
            # MAGIC
            return ABSTAIN
    return ABSTAIN


def event_patterns_helper(x, rules, general_location=False):
    # find best matching pattern and use corresponding rule (slots) to return role label
    # need to find range of match
    if x.mixed_ner_spans and x.mixed_ner:
        trigger_spans = x.mixed_ner_spans[x.trigger_idx]
        argument_spans = x.mixed_ner_spans[x.argument_idx]

        # Change formatting to match the formatting used for the converter rule slots
        trigger_entity_type = x.trigger['entity_type'].upper()
        argument_entity_type = x.argument['entity_type'].upper().replace('-', '_')

        if general_location and argument_entity_type in location_subtypes:
            argument_entity_type = 'LOCATION'

        best_rule, best_match = find_best_pattern_match(x.mixed_ner, rules, trigger_spans, argument_spans)

        if best_rule and best_match:
            """
            Idea here is to match the slots of the best rule with the trigger and argument via entity type and
            their position in the subset of entities that are within the spans of the best match.
            Position here refers to the n-th occurrence of the entity type within the entities. 
            """
            # within span of the best match
            entities_subset = [entity for span, entity in zip(x.mixed_ner_spans, x.entities) if
                               span[0] >= best_match.start() and span[1] <= best_match.end()]
            trigger_position = next(
                (idx for idx, entity in enumerate(entities_subset) if entity['id'] == x.trigger['id']), None)
            argument_position = next(
                (idx for idx, entity in enumerate(entities_subset) if entity['id'] == x.argument['id']), None)

            entity_types = []
            for entity in entities_subset:
                entity_type = entity['entity_type'].upper().replace('-', '_')
                if general_location and entity_type in location_subtypes:
                    entity_type = 'LOCATION'
                entity_types.append(entity_type)

            trigger_pos = entity_types[:trigger_position + 1].count(trigger_entity_type)
            argument_pos = entity_types[:argument_position + 1].count(argument_entity_type)

            trigger_match = False
            argument_match = False
            role = ''
            # obv check beforehand if sample_trigger_position and sample_argument_position are None, ABSTAIN
            # or check during search for best pattern
            for slot in best_rule.slots:
                if slot.entity_type == trigger_entity_type and slot.position == trigger_pos:
                    trigger_match = True
                if slot.entity_type == argument_entity_type and slot.position == argument_pos:
                    role = slot.role
                    argument_match = True
            if trigger_match and argument_match and role:
                assert role in labels
                return labels[role]  # mapping to correct label index

    return ABSTAIN


@labeling_function(resources=dict(rules=original_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns(x, rules):
    return event_patterns_helper(x, rules)


@labeling_function(resources=dict(rules=original_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_route_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=False)
    if label == route:
        return route
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_location(x, rules):
    return event_patterns_helper(x, rules, general_location=True)


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_location_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == location:
        return location
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_direction_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == direction:
        return direction
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_delay_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == delay:
        return delay
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_startloc_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    arg_entity_type = x.argument['entity_type']
    if label == start_loc and arg_entity_type != 'location_route':
        return start_loc
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_endloc_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    arg_entity_type = x.argument['entity_type']
    if label == end_loc and arg_entity_type != 'location_route':
        return end_loc
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_startdate_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == start_date:
        return start_date
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_enddate_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == end_date:
        return end_date
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_cause_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == cause:
        return cause
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[pre_trigger_idx, pre_argument_idx])
def lf_event_patterns_general_jamlength_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == jam_length:
        return jam_length
    else:
        return ABSTAIN
