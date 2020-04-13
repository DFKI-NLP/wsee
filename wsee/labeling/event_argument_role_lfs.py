from typing import Any

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
def check_required_args(entitiy_freqs):
    if any(loc_type in entitiy_freqs
           for loc_type in ['location', 'location_route', 'location_street', 'location_stop', 'location_city']) and \
            'trigger' in entitiy_freqs:
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

def lf_location(x, same_sentence=True, nearest=True, check_event_type=True,
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
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if same_sentence:
        if lf_somajo_separate_sentence(x) != ABSTAIN:
            return ABSTAIN
    if nearest:
        if same_sentence:
            if not is_nearest_trigger(x.between_distance, x.sentence_trigger_distances):
                return ABSTAIN
        else:
            if not is_nearest_trigger(x.between_distance, x.all_trigger_distances):
                return ABSTAIN

    arg_entity_type = x.argument['entity_type']
    if arg_entity_type not in entity_types:
        return ABSTAIN

    if not check_event_type:
        return location
    else:
        for event_class, location_types in event_type_location_type_map.items():
            if arg_entity_type in location_types and event_type_lf_map[event_class](x) == event_class:
                return location
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_somajo_doc, get_between_distance,
                        get_all_trigger_distances, get_sentence_trigger_distances])
def lf_location_same_sentence_is_event(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and lf_direction_type(x) == ABSTAIN:
        return lf_location(x, nearest=False)
    else:
        return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_somajo_doc, get_between_distance,
                        get_all_trigger_distances, get_sentence_trigger_distances])
def lf_location_same_sentence_nearest_is_event(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and lf_direction_type(x) == ABSTAIN:
        return lf_location(x)
    else:
        return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_somajo_doc, get_between_distance,
                        get_all_trigger_distances, get_sentence_trigger_distances])
def lf_location_adjacent(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and \
            lf_direction_type(x) == ABSTAIN and \
            lf_multiple_same_event_type(x) == ABSTAIN and lf_somajo_separate_sentence(x) == ABSTAIN \
            and x.between_distance < 3:
        return location
    return ABSTAIN


# delay role
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_somajo_doc, get_between_distance,
                        get_all_trigger_distances, get_sentence_trigger_distances, get_argument_right_tokens,
                        get_argument_left_tokens])
def lf_delay_event_sentence(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if 'früher' in x.argument_right_tokens[:2]:
        return no_arg
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['duration']:
            if event_trigger_lfs.lf_delay_cat(x) == event_trigger_lfs.Delay or \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam or \
                    event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction or \
                    event_trigger_lfs.lf_accident_context(x) == event_trigger_lfs.Accident:
                if lf_somajo_separate_sentence(x) == ABSTAIN or 'Zeitverlust' in x.argument_left_tokens[-4:]:
                    return delay
    return ABSTAIN


# direction role
# TODO: no location_route, location_street and check for event types
#  Accident (loc, loc_city), CanceledRoute (loc_stop, loc_city), CanceledStop (none), Delay (loc_stop),
#  Obstruction (loc, loc_city), RailReplacementService (loc_stop), TrafficJam (loc, loc_city)
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_somajo_doc])
def lf_direction_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location', 'location_city', 'location_stop', 'location_street']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
                if any(token.lower() in ['nach', 'richtung'] for token in x.argument_left_tokens[-1:]) or \
                        'Ankunft' in x.argument_left_tokens[-4:] or \
                        'richtung' in x.argument['text'].lower() or 'stadteinwärts' in x.argument['text'].lower() or \
                        'stadtauswärts' in x.argument['text'].lower():
                    return direction
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_left_ner,
                        get_somajo_doc])
def lf_loc_stop_direction_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location_stop']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                    (any(token.lower() in ['nach', 'richtung'] for token in x.argument_left_tokens[-1:]) or
                     'Ankunft' in x.argument_left_tokens[-4:] or
                     ('argument_left_ner' in x and x.argument_left_ner and
                      x.argument_left_ner[-1][2:] == 'LOCATION_ROUTE')) and \
                    (event_trigger_lfs.lf_canceledroute_cat(x) == event_trigger_lfs.CanceledRoute or
                     event_trigger_lfs.lf_delay_cat(x) == event_trigger_lfs.Delay or
                     event_trigger_lfs.lf_railreplacementservice_cat(x) == event_trigger_lfs.RailReplacementService):
                return direction
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_left_ner,
                        get_somajo_doc])
def lf_loc_loc_city_direction_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location', 'location_city', 'location_stop']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and \
                    (any(token.lower() in ['nach', 'richtung'] for token in x.argument_left_tokens[-1:]) or
                     'Ankunft' in x.argument_left_tokens[-4:] or
                     'richtung' in x.argument['text'].lower() or 'stadteinwärts' in x.argument['text'].lower() or
                     'stadtauswärts' in x.argument['text'].lower() or
                     ('argument_left_ner' in x and x.argument_left_ner and
                      x.argument_left_ner[-1][2:] == 'LOCATION_ROUTE')) and \
                    (event_trigger_lfs.lf_accident_context(x) == event_trigger_lfs.Accident or
                     event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction or
                     event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam):
                return direction
    return ABSTAIN


# start_loc
# TODO: no location_route and check for event types
#  Accident (loc, loc_city), CanceledRoute (loc_stop), CanceledStop (none), Delay (loc_stop),
#  Obstruction (all), RailReplacementService (loc_stop), TrafficJam (loc, loc_city, loc_street)
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_right_tokens,
                        get_argument_right_ner, get_argument_left_ner, get_somajo_doc])
def lf_start_location_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            event_trigger_lfs.canceledstop_keywords(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                (any(token.lower() in ['zw.', 'zwischen', 'ab', 'von', 'einfahrt', 'anschlussstelle']
                     for token in x.argument_left_tokens[-2:]) or
                 (x.argument_right_tokens and x.argument_right_tokens[0] in ['-', '<', '>', '<>'] and
                  'argument_right_ner' in x and len(x.argument_right_ner) > 1 and
                  x.argument_right_ner[1][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP'])):
            if 'von' == x.argument_left_tokens[-1:] and \
                  'argument_left_ner' in x and len(x.argument_left_ner) > 1 and \
                    x.argument_left_ner[-2][2:] in ['LOCATION_ROUTE']:
                return ABSTAIN
            else:
                return start_loc
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_right_tokens,
                        get_argument_right_ner, get_argument_left_ner, get_somajo_doc, get_between_distance,
                        get_sentence_trigger_distances])
def lf_start_location_questionable(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            event_trigger_lfs.canceledstop_keywords(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        if is_nearest_trigger(x.between_distance, x.sentence_trigger_distances) and lf_not_an_event(x) == ABSTAIN and \
                (any(token.lower() in ['bei', 'höhe'] for token in x.argument_left_tokens[-2:])):
            return start_loc
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_right_tokens,
                        get_argument_right_ner, get_argument_left_ner, get_somajo_doc, get_between_distance,
                        get_sentence_trigger_distances])
def lf_start_location_nearest(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            event_trigger_lfs.canceledstop_keywords(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        if is_nearest_trigger(x.between_distance, x.sentence_trigger_distances) and lf_not_an_event(x) == ABSTAIN and \
                (any(token.lower() in ['zw.', 'zwischen', 'ab', 'von', 'einfahrt', 'anschlussstelle']
                     for token in x.argument_left_tokens[-1:]) or
                 (x.argument_right_tokens and x.argument_right_tokens[0] in ['-', '<', '>', '<>'] and
                  'argument_right_ner' in x and len(x.argument_right_ner) > 1 and
                  x.argument_right_ner[1][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP'])):
            if 'von' == x.argument_left_tokens[-1:] and \
                  'argument_left_ner' in x and len(x.argument_left_ner) > 1 and \
                  x.argument_left_ner[-2][2:] in ['LOCATION_ROUTE']:
                return ABSTAIN
            else:
                return start_loc
    return ABSTAIN


# end_loc
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_left_ner,
                        get_somajo_doc, get_argument_right_ner])
def lf_end_location_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or \
            event_trigger_lfs.canceledstop_keywords(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                ((any(token.lower() in ['zw.', 'zwischen'] for token in x.argument_left_tokens[-6:]) and
                  any(token.lower() in ['und', 'u.', '<', '>', '<>'] for token in x.argument_left_tokens[-1:])) or
                 (any(token.lower() in ['von'] for token in x.argument_left_tokens[-6:-2]) and
                  any(token.lower() in ['auf'] for token in x.argument_left_tokens[-3:])) or
                 any(token.lower() in ['bis'] for token in x.argument_left_tokens[-1:]) or
                 (x.argument_left_tokens and '-' == x.argument_left_tokens[-1] and
                  'argument_right_ner' in x and len(x.argument_right_ner) > 1 and
                  x.argument_right_ner[-2][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP'])):
            return end_loc
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_left_ner,
                        get_somajo_doc, get_between_distance, get_sentence_trigger_distances, get_argument_right_ner])
def lf_end_location_nearest(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        if is_nearest_trigger(x.between_distance, x.sentence_trigger_distances) and lf_not_an_event(x) == ABSTAIN and \
                ((any(token.lower() in ['zw.', 'zwischen'] for token in x.argument_left_tokens[-6:]) and
                  any(token.lower() in ['und', 'u.', '<', '>', '<>'] for token in x.argument_left_tokens[-1:])) or
                 (any(token.lower() in ['von'] for token in x.argument_left_tokens[-6:-2]) and
                  any(token.lower() in ['auf'] for token in x.argument_left_tokens[-3:])) or
                 any(token.lower() in ['bis'] for token in x.argument_left_tokens[-1:]) or
                 (x.argument_left_tokens and '-' == x.argument_left_tokens[-1] and
                  'argument_right_ner' in x and len(x.argument_right_ner) > 1 and
                  x.argument_right_ner[-2][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY', 'LOCATION_STOP'])):
            return end_loc
    return ABSTAIN


# start_date
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_right_tokens,
                        get_somajo_doc, get_argument_right_ner])
def lf_start_date_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or 'Meldung' in x.argument_right_tokens:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                    ((any(token.lower() in ['ab', 'von', 'vom'] for token in x.argument_left_tokens[-3:]) and
                      not any(token.lower() in ['bis'] for token in x.argument_left_tokens[-3:])) or
                     (x.argument_right_tokens and x.argument_right_tokens[0] in ['und', '/', '-'] and
                      'argument_right_ner' in x and len(x.argument_right_ner) > 1 and
                      x.argument_right_ner[1][2:] in ['DATE', 'TIME'])):
                return start_date
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_right_tokens,
                        get_somajo_doc, get_between_distance, get_sentence_trigger_distances,
                        get_entity_trigger_distances])
def lf_start_date_nearest(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or 'Meldung' in x.argument_right_tokens:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            if is_nearest_trigger(x.between_distance, x.sentence_trigger_distances) and \
                    x.entity_trigger_distances[arg_entity_type] and \
                    x.between_distance <= min(x.entity_trigger_distances[arg_entity_type]) and \
                    lf_not_an_event(x) == ABSTAIN and lf_end_date_type(x) == ABSTAIN:
                return start_date
    return ABSTAIN


# end_date
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_right_tokens,
                        get_somajo_doc, get_argument_left_ner])
def lf_end_date_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN or 'Meldung' in x.argument_right_tokens:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                    ((any(token.lower() in ['bis', 'endet'] for token in x.argument_left_tokens[-3:]) and
                      not any(token.lower() in ['von'] for token in x.argument_left_tokens[-2:])) or
                     (x.argument_left_tokens and x.argument_left_tokens[-1] in ['und', '/', '-'] and
                      'argument_left_ner' in x and len(x.argument_left_ner) > 1 and
                      x.argument_left_ner[1][2:] in ['DATE', 'TIME'])):
                return end_date
    return ABSTAIN


# cause
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_argument_left_tokens, get_argument_right_tokens,
                        get_somajo_doc])
def lf_cause_type(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['trigger']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                    (event_trigger_lfs.check_cause_keywords(x.argument_left_tokens[-4:], x) or
                     (x.argument_right_tokens and x.argument_right_tokens[0].lower() in ['erzeugt', 'erzeugen'])):
                # TODO check trigger-arg order, some event types have higher priority
                #  often Accident cause for TrafficJam
                return cause
    return ABSTAIN


@labeling_function(resources=dict(cause_mapping=traffic_event_causes), pre=[get_trigger_text, get_argument_text])
def lf_cause_gaz_file(x, cause_mapping):
    # TODO add more cause pairs, coverage is 0%
    if x.argument_text in cause_mapping:
        if cause_mapping[x.argument_text] == x.trigger_text:
            return cause
    return ABSTAIN


# jam_length
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_somajo_doc, get_between_distance,
                        get_all_trigger_distances, get_sentence_trigger_distances, get_entity_trigger_distances])
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


@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_somajo_doc, get_between_distance,
                        get_all_trigger_distances, get_sentence_trigger_distances, get_entity_trigger_distances])
def lf_distance_nearest(x):
    if lf_too_far_40(x) != ABSTAIN or lf_multiple_same_event_type(x) != ABSTAIN:
        return ABSTAIN
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['distance']:
            if is_nearest_trigger(x.between_distance, x.sentence_trigger_distances) and \
                    x.entity_trigger_distances[arg_entity_type] and \
                    x.between_distance <= min(x.entity_trigger_distances[arg_entity_type]) and \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam:
                return jam_length
    return ABSTAIN


# route
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs, get_argument, get_somajo_doc, get_between_distance,
                        get_all_trigger_distances, get_sentence_trigger_distances])
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


# no_args
@labeling_function(pre=[get_trigger, get_between_tokens, get_trigger_left_tokens, get_trigger_right_tokens,
                        get_entity_type_freqs])
def lf_not_an_event(x):
    if event_trigger_lfs.lf_negative(x) == event_trigger_lfs.O:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens, get_argument, get_spacy_doc])
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


@labeling_function(pre=[get_trigger, get_between_tokens, get_argument, get_stanford_doc])
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


@labeling_function(pre=[get_trigger, get_between_tokens, get_argument, get_somajo_doc])
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
        if sentence_start <= m_start + tolerance and m_end <= sentence_end + tolerance and \
                x.somajo_doc['trigger'] in sentence and x.somajo_doc['argument'] in sentence:
            same_sentence = True
            break
    if same_sentence:
        return ABSTAIN
    else:
        return no_arg


@labeling_function(
    pre=[get_trigger, get_between_tokens, get_argument, get_between_distance, get_all_trigger_distances])
def lf_not_nearest_event(x):
    if is_nearest_trigger(x.between_distance, x.all_trigger_distances):
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[get_trigger, get_between_tokens, get_argument, get_between_distance,
                        get_sentence_trigger_distances])
def lf_not_nearest_same_sentence_event(x):
    if is_nearest_trigger(x.between_distance, x.sentence_trigger_distances):
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[get_trigger, get_between_tokens, get_argument, get_between_distance,
                        get_sentence_trigger_distances, get_entity_trigger_distances])
def lf_not_nearest_event_argument(x):
    argument_type = x.argument['entity_type']
    if argument_type not in ['location', 'location_city', 'location_route', 'location_stop', 'location_street', 'date',
                             'time', 'duration', 'distance', 'trigger']:
        return no_arg
    if is_nearest_trigger(x.between_distance, x.sentence_trigger_distances) and \
            'entity_trigger_distances' in x and \
            x.entity_trigger_distances[argument_type] and \
            x.between_distance <= min(x.entity_trigger_distances[argument_type]):
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[get_trigger, get_between_tokens, get_argument, get_between_distance])
def lf_overlapping(x):
    if x.between_distance < 0:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[get_between_distance])
def lf_too_far_40(x):
    if x.between_distance > 40:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[get_trigger, get_between_tokens])
def lf_multiple_same_event_type(x):
    # TODO check and use in all lfs
    #  also use keyword list and loop over them instead of trying to exactly match the trigger text
    trigger_text = x.trigger['text']
    if trigger_text in x.between_tokens:
        return no_arg
    else:
        return ABSTAIN


# general
@labeling_function(pre=[get_trigger, get_between_tokens, get_argument, get_spacy_doc])
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
                (idx for idx, entity in enumerate(entities_subset) if entity['id'] == x.trigger_id), None)
            argument_position = next(
                (idx for idx, entity in enumerate(entities_subset) if entity['id'] == x.argument_id), None)

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


@labeling_function(resources=dict(rules=original_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                              get_argument, get_argument_idx,
                                                              get_mixed_ner])
def lf_event_patterns(x, rules):
    return event_patterns_helper(x, rules)


@labeling_function(resources=dict(rules=original_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                              get_argument, get_argument_idx,
                                                              get_mixed_ner])
def lf_event_patterns_route_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=False)
    if label == route:
        return route
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_location(x, rules):
    return event_patterns_helper(x, rules, general_location=True)


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_location_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == location:
        return location
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_direction_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == direction:
        return direction
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_delay_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == delay:
        return delay
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_startloc_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    arg_entity_type = x.argument['entity_type']
    if label == start_loc and arg_entity_type != 'location_route':
        return start_loc
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_endloc_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    arg_entity_type = x.argument['entity_type']
    if label == end_loc and arg_entity_type != 'location_route':
        return end_loc
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_startdate_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == start_date:
        return start_date
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_enddate_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == end_date:
        return end_date
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_cause_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == cause:
        return cause
    else:
        return ABSTAIN


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_between_tokens, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_jamlength_type(x, rules):
    label = event_patterns_helper(x, rules, general_location=True)
    if label == jam_length:
        return jam_length
    else:
        return ABSTAIN
