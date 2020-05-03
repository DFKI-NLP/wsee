from fuzzywuzzy import process
from snorkel.labeling import labeling_function

from wsee.labeling import event_trigger_lfs
from wsee.preprocessors.pattern_event_processor import parse_pattern_file, find_best_pattern_match, location_subtypes
from wsee.preprocessors.preprocessors import *
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
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type not in entity_types:
        return ABSTAIN

    between_distance = x.between_distance
    all_trigger_distances = get_all_trigger_distances(x)
    if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
            (x.trigger['text'] in ['aus', 'aus.']):
        return ABSTAIN
    if same_sentence:
        if x.separate_sentence:
            return ABSTAIN
    if nearest:
        if not is_nearest_trigger(between_distance, all_trigger_distances):
            return ABSTAIN

    if not check_event_type or x.arg_location_type_event_type_match:
        return location
    return ABSTAIN


@labeling_function(pre=[])
def lf_location_same_sentence_is_event(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and lf_direction_type(x) == ABSTAIN:
        return lf_location(x, nearest=False)
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_location_same_sentence_nearest_is_event(x):
    if lf_start_location_type(x) == ABSTAIN and lf_end_location_type(x) == ABSTAIN and lf_direction_type(x) == ABSTAIN:
        return lf_location(x, nearest=True)
    else:
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
    between_distance = x.between_distance
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
    between_distance = x.between_distance
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


def get_first_of_entity_types(entities, entity_types: List[str], exception_list: List[str] = None):
    """
    :param entities: List of entities.
    :param entity_types: List of desired entity_types.
    :param exception_list: List of exceptions.
    :return: First entity, whose entity type is in entity_types and whose text is not in the exception_list.
    """
    first_entity = None
    for entity in entities:
        if entity['entity_type'] in entity_types and \
                (first_entity is None or first_entity['start'] > entity['start']):
            if exception_list and entity['text'] in exception_list:
                continue
            else:
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
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['duration']:
            argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
            argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                    (x.separate_sentence and 'Zeitverlust' not in argument_left_tokens[-4:]):
                return ABSTAIN
            if 'früher' in argument_right_tokens[:2]:
                return ABSTAIN
            if event_trigger_lfs.lf_delay_cat(x) == event_trigger_lfs.Delay or \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam or \
                    event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction or \
                    event_trigger_lfs.lf_accident_context(x) == event_trigger_lfs.Accident:
                return delay
    return ABSTAIN


# delay role
@labeling_function(pre=[])
def lf_delay_event_sentence_check(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['duration']:
            argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
            argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                    (x.separate_sentence and 'Zeitverlust' not in argument_left_tokens[-4:]):
                return ABSTAIN
            if 'früher' in argument_right_tokens[:2]:
                return ABSTAIN
            if event_trigger_lfs.lf_delay_cat(x) == event_trigger_lfs.Delay or \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam or \
                    event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction or \
                    event_trigger_lfs.lf_accident_context(x) == event_trigger_lfs.Accident:
                if get_entity_distance(x.trigger, x.argument) > 0 and \
                        x.argument['text'].islower() and \
                        x.argument['start'] < x.trigger['start']:
                    # Rationale here: duration can only occur after a lower case trigger,
                    # e.g. "verspätet sich um 15 min"
                    # Only case, where a lower case trigger can occur after the duration "15 später", they are
                    # adjacent
                    return ABSTAIN
                else:
                    return delay
    return ABSTAIN


@labeling_function(pre=[])
def lf_delay_earlier_negative(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['duration']:
            argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
            argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                    (x.separate_sentence and 'Zeitverlust' not in argument_left_tokens[-4:]):
                return ABSTAIN
            if 'früher' in argument_right_tokens[:2]:
                return no_arg
    return ABSTAIN


# direction role
def lf_direction(x, order=True):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location', 'location_city', 'location_stop', 'location_street']:
            argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or x.separate_sentence or x.not_an_event:
                return ABSTAIN
            if lf_start_location_type(x) != ABSTAIN or lf_end_location_type(x) != ABSTAIN:
                return ABSTAIN
            if order and x.argument['start'] > x.trigger['start']:
                return ABSTAIN
            article_preposition_offset = get_article_preposition_offset(argument_left_tokens)

            if any(token.lower() in ['nach', 'richtung', 'fahrtrichtung', '->']
                   for token in argument_left_tokens[-1 - article_preposition_offset:]) or \
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


def has_direction_markers(argument_text, argument_left_tokens, article_preposition_offset=0):
    return any(token.lower() in ['nach', 'richtung', 'fahrtrichtung', '->']
               for token in argument_left_tokens[-1 - article_preposition_offset:]) or \
           argument_text.lower() in ['richtung', 'richtungen', 'stadteinwärts', 'stadtauswärts',
                                     'beide richtungen', 'beiden richtungen', 'gegenrichtung',
                                     'je richtung']


@labeling_function(pre=[])
def lf_direction_type(x):
    return lf_direction(x, order=False)


@labeling_function(pre=[])
def lf_direction_order(x):
    return lf_direction(x, order=True)


# start_loc
def lf_start_location(x, nearest=False):
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
        argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
        argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
        argument_right_ner = get_windowed_right_ner(x.argument, x.ner_tags)
        if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                event_trigger_lfs.lf_canceledstop_cat(x) == event_trigger_lfs.CanceledStop:
            return ABSTAIN
        if x.separate_sentence or x.not_an_event:
            return ABSTAIN
        if nearest:
            between_distance = x.between_distance
            all_trigger_distances = get_all_trigger_distances(x)
            if not is_nearest_trigger(between_distance, all_trigger_distances):
                return ABSTAIN
        if has_start_location_markers(argument_left_tokens, argument_right_tokens,
                                      argument_left_ner, argument_right_ner):
            return start_loc
    return ABSTAIN


def has_start_location_markers(argument_left_tokens, argument_right_tokens,
                               argument_left_ner, argument_right_ner):
    indicator_idx, match_token = next((idx, token for idx, token in enumerate(argument_left_tokens[-3:])
                                       if token.lower() in ['zw', 'zw.', 'zwischen', 'ab', 'von']), (-1, None))
    entity_between_indicator = indicator_idx > -1 and any(left_ner[2:]
                                                          in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY',
                                                              'LOCATION_STOP', 'LOCATION_ROUTE']
                                                          for left_ner in argument_left_ner[-3 + indicator_idx:])
    if match_token and match_token.lower() == 'von' and 'bis' not in argument_right_tokens:
        # To avoid cases where 'von' indicates a start location of a train line, but not the start location of an event
        return False
    end_loc_after_symbol = argument_right_tokens and argument_right_tokens[0] in ['-', '<', '>', '<>'] and \
                           len(argument_right_ner) > 1 and \
                           argument_right_ner[1][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY',
                                                         'LOCATION_STOP', 'LOCATION_ROUTE']
    if (indicator_idx > -1 and not entity_between_indicator) or end_loc_after_symbol:
        return True
    else:
        return False


@labeling_function(pre=[])
def lf_start_location_type(x):
    return lf_start_location(x, nearest=False)


@labeling_function(pre=[])
def lf_start_location_nearest(x):
    return lf_start_location(x, nearest=True)


# end_loc
def lf_end_location(x, nearest=False):
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_city', 'location_stop']:
        argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
        argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
        between_distance = x.between_distance
        sentence_trigger_distances = get_sentence_trigger_distances(x)
        if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or x.separate_sentence or x.not_an_event or \
                event_trigger_lfs.lf_canceledstop_cat(x) != ABSTAIN:
            return ABSTAIN
        if nearest and not is_nearest_trigger(between_distance, sentence_trigger_distances):
            return ABSTAIN

        if len(argument_left_tokens) > 2 and argument_left_tokens[-3].lower() in ['nach', 'richtung', 'fahrtrichtung',
                                                                                  '->'] \
                and argument_left_tokens[-1] == '-':
            return ABSTAIN
        if len(argument_left_tokens) > 2 and argument_left_tokens[-1] == '-':
            # Avoid patterns like: "A1 Münster - Köln in beiden Richtungen ...", where Köln is a direction
            if argument_left_ner[-3][2:] == 'LOCATION_STREET' and \
                    argument_left_ner[-2][2:] in ['LOCATION', 'LOCATION_CITY']:
                return ABSTAIN
        if has_end_loc_prefix(argument_left_tokens) or has_preceding_start_loc(argument_left_tokens, argument_left_ner):
            return end_loc
    return ABSTAIN


def has_end_loc_prefix(argument_left_tokens):
    article_preposition_offset = get_article_preposition_offset(argument_left_tokens)
    return any(token.lower() in ['bis'] for token in argument_left_tokens[-1 - article_preposition_offset:])


def has_preceding_start_loc(argument_left_tokens, argument_left_ner):
    article_preposition_offset = get_article_preposition_offset(argument_left_tokens)
    preceding_start_loc = any(token.lower() in ['zw.', 'zwischen'] for token in argument_left_tokens[-6:-1])
    concatenation_prefix = any(token.lower() in ['und', 'u.', '<', '>', '<>', '&']
                               for token in argument_left_tokens[-1 - article_preposition_offset:])
    hyphenated_start_end_pair = argument_left_tokens and '-' == argument_left_tokens[-1] and \
                                len(argument_left_ner) > 1 and \
                                argument_left_ner[-2][2:] in ['LOCATION', 'LOCATION_STREET', 'LOCATION_CITY',
                                                              'LOCATION_STOP']
    return (preceding_start_loc and concatenation_prefix) or hyphenated_start_end_pair


def get_article_preposition_offset(argument_left_tokens):
    article_preposition_offset = 0
    if argument_left_tokens and argument_left_tokens[-1] in ['der', 'des', 'dem', 'den', 'zu', 'zur', 'zum']:
        article_preposition_offset = 1
    return article_preposition_offset


@labeling_function(pre=[])
def lf_end_location_type(x):
    return lf_end_location(x, nearest=False)


@labeling_function(pre=[])
def lf_end_location_nearest(x):
    return lf_end_location(x, nearest=True)


# start_date
@labeling_function(pre=[])
def lf_start_date_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
            argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
            argument_right_ner = get_windowed_right_ner(x.argument, x.ner_tags)
            between_tokens = get_between_tokens(x)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or x.separate_sentence or x.not_an_event:
                return ABSTAIN
            elif (any(left_token.lower() == 'gültig' for left_token in argument_left_tokens[-4:]) and
                  'ab' in argument_left_tokens[-3:]) or 'Meldung' in between_tokens:
                return ABSTAIN
            elif has_start_date_markers(argument_left_tokens, argument_right_tokens, argument_right_ner):
                return start_date
    return ABSTAIN


def has_start_date_markers(argument_left_tokens, argument_right_tokens, argument_right_ner):
    if ((any(token.lower() in ['ab', 'von', 'vom'] for token in argument_left_tokens[-3:]) and
         not any(token.lower() in ['bis'] for token in argument_left_tokens[-3:])) or
            (argument_right_tokens and argument_right_tokens[0] in ['und', '/', '-', '->'] and
             len(argument_right_ner) > 1 and argument_right_ner[1][2:] in ['DATE', 'TIME'])):
        return True
    else:
        return False


@labeling_function(pre=[])
def lf_start_date_first(x):
    first_date = get_first_of_entity_types(x.entities, ['date', 'time'])
    if check_required_args(x.entity_type_freqs) and first_date and first_date['id'] == x.argument['id']:
        argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
        between_tokens = get_between_tokens(x)
        if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or x.not_an_event or \
                x.separate_sentence:
            return ABSTAIN
        elif 'Meldung' in between_tokens or \
                (any(left_token.lower() == 'gültig' for left_token in argument_left_tokens[-4:]) and
                 'ab' in argument_left_tokens[-3:]):
            return ABSTAIN
        elif lf_end_date_type(x) != ABSTAIN:
            return ABSTAIN
        else:
            return start_date
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_start_date_adjacent(x):
    between_distance = x.between_distance
    if check_required_args(x.entity_type_freqs) and x.argument['entity_type'] in ['date', 'time'] \
            and between_distance < 3:
        argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
        between_tokens = get_between_tokens(x)
        if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or x.not_an_event or \
                x.separate_sentence:
            return ABSTAIN
        elif 'Meldung' in between_tokens or \
                (any(left_token.lower() == 'gültig' for left_token in argument_left_tokens[-4:]) and
                 'ab' in argument_left_tokens[-3:]):
            return ABSTAIN
        else:
            argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
            if has_end_date_markers(argument_left_tokens, argument_left_ner):
                return ABSTAIN
            else:
                return start_date
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_date_negative(x):
    if check_required_args(x.entity_type_freqs) and x.argument['entity_type'] in ['date', 'time']:
        argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
        between_tokens = get_between_tokens(x)
        if 'Meldung' in between_tokens or \
                (any(left_token.lower() == 'gültig' for left_token in argument_left_tokens[-4:]) and
                 'ab' in argument_left_tokens[-3:]):
            return no_arg
        else:
            return ABSTAIN
    return ABSTAIN


# end_date
@labeling_function(pre=[])
def lf_end_date_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
            argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
            argument_left_ner = get_windowed_left_ner(x.argument, x.ner_tags)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or x.separate_sentence or x.not_an_event:
                return ABSTAIN
            elif 'Meldung' in argument_right_tokens:
                return ABSTAIN
            elif has_end_date_markers(argument_left_tokens, argument_left_ner):
                return end_date
    return ABSTAIN


def has_end_date_markers(argument_left_tokens, argument_left_ner):
    if ((any(token.lower() in ['bis', 'endet', 'enden'] for token in argument_left_tokens[-3:]) and
         not any(token.lower() in ['von', 'ab', 'vom'] for token in argument_left_tokens[-2:])) or
            (argument_left_tokens and argument_left_tokens[-1] in ['und', '/', '-'] and
             len(argument_left_ner) > 1 and argument_left_ner[1][2:] in ['DATE', 'TIME'])):
        return True
    else:
        return False


# cause
@labeling_function(pre=[])
def lf_cause_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['trigger', 'event_cause']:
            argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
            argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                    x.separate_sentence or x.not_an_event:
                return ABSTAIN
            if (event_trigger_lfs.check_cause_keywords(argument_left_tokens[-4:], x) or
                    (argument_right_tokens and argument_right_tokens[0].lower() in ['erzeugt', 'erzeugen'])):
                # TODO check trigger-arg order, some event types have higher priority
                #  often Accident cause for TrafficJam
                return cause
    return ABSTAIN


@labeling_function(pre=[])
def lf_cause_order(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['trigger', 'event_cause']:
            argument_left_tokens = get_windowed_left_tokens(x.argument, x.tokens)
            argument_right_tokens = get_windowed_right_tokens(x.argument, x.tokens)
            between_distance = get_entity_distance(x.argument, x.trigger)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                    x.separate_sentence or x.not_an_event:
                return ABSTAIN
            if x.argument['start'] < x.trigger['start']:
                # E.g.: "wegen Unfall gesperrt", but "gesperrt nach Unfall"
                return ABSTAIN
            if (event_trigger_lfs.check_cause_keywords(argument_left_tokens[-4:], x) or
                    (argument_right_tokens and argument_right_tokens[0].lower() in ['erzeugt', 'erzeugen'])):
                return cause
            elif between_distance < 5 and \
                    (event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam or
                     event_trigger_lfs.lf_obstruction_cat(x) == event_trigger_lfs.Obstruction):
                # Accidents are often causes for obstructions/ traffic jams
                highest = process.extractOne(x.argument['text'], event_trigger_lfs.accident_keywords)
                if highest[1] >= 90:
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
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['distance']:
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                    x.separate_sentence:
                return ABSTAIN
            elif event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam:
                return jam_length
    return ABSTAIN


@labeling_function(pre=[])
def lf_distance_nearest(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['distance']:
            between_distance = x.between_distance
            sentence_trigger_distances = get_sentence_trigger_distances(x)
            entity_trigger_distances = get_entity_trigger_distances(x)
            if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type:
                return ABSTAIN
            elif is_nearest_trigger(between_distance, sentence_trigger_distances) and \
                    entity_trigger_distances[arg_entity_type] and \
                    between_distance <= min(entity_trigger_distances[arg_entity_type]) and \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam:
                return jam_length
    return ABSTAIN


# route
@labeling_function(pre=[])
def lf_route_type(x):
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location_route']:
        if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                x.separate_sentence:
            return ABSTAIN
        elif event_trigger_lfs.lf_canceledstop_cat(x) == event_trigger_lfs.CanceledStop:
            return route
    return ABSTAIN


@labeling_function(pre=[])
def lf_route_type_order(x):
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location_route']:
        if lf_too_far_40(x) == no_arg or x.is_multiple_same_event_type or \
                x.separate_sentence:
            return ABSTAIN
        if event_trigger_lfs.lf_canceledstop_cat(x) == event_trigger_lfs.CanceledStop and \
                x.argument['start'] < x.trigger['start']:
            return route
    return ABSTAIN


# no_args
@labeling_function(pre=[])
def lf_not_an_event(x):
    if x.not_an_event:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_somajo_separate_sentence(x):
    if x.separate_sentence:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_not_nearest_event(x):
    between_distance = x.between_distance
    all_trigger_distances = get_all_trigger_distances(x)
    if is_nearest_trigger(between_distance, all_trigger_distances):
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[])
def lf_not_nearest_same_sentence_event(x):
    between_distance = x.between_distance
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    if is_nearest_trigger(between_distance, sentence_trigger_distances):
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[])
def lf_not_nearest_event_argument(x):
    between_distance = x.between_distance
    sentence_trigger_distances = get_sentence_trigger_distances(x)
    entity_trigger_distances = get_entity_trigger_distances(x)
    argument_type = x.argument['entity_type']
    if argument_type not in ['location', 'location_city', 'location_route', 'location_stop', 'location_street', 'date',
                             'time', 'duration', 'distance', 'trigger', 'event_cause']:
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
    between_distance = x.between_distance
    if between_distance < 0:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_too_far_40(x):
    between_distance = x.between_distance
    if between_distance > 40:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_multiple_same_event_type(x):
    if x.is_multiple_same_event_type:
        return no_arg
    else:
        return ABSTAIN


def event_patterns_helper(x, rules, trigger_idx, argument_idx, general_location=False):
    # find best matching pattern and use corresponding rule (slots) to return role label
    # need to find range of match
    if x.mixed_ner_spans and x.mixed_ner:
        trigger_spans = x.mixed_ner_spans[trigger_idx]
        argument_spans = x.mixed_ner_spans[argument_idx]

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


@labeling_function(resources=dict(rules=original_rules), pre=[])
def lf_event_patterns(x, rules):
    trigger: Dict[str, Any] = x.trigger
    trigger_idx: int = get_entity_idx(trigger['id'], x.entities)
    argument: Dict[str, Any] = x.argument
    argument_idx: int = get_entity_idx(argument['id'], x.entities)
    return event_patterns_helper(x, rules, trigger_idx, argument_idx, general_location=False)


@labeling_function(resources=dict(rules=general_location_rules), pre=[])
def lf_event_patterns_general_location(x, rules):
    trigger: Dict[str, Any] = x.trigger
    trigger_idx: int = get_entity_idx(trigger['id'], x.entities)
    argument: Dict[str, Any] = x.argument
    argument_idx: int = get_entity_idx(argument['id'], x.entities)
    label = event_patterns_helper(x, rules, trigger_idx, argument_idx, general_location=True)
    if label == route:
        return ABSTAIN
    else:
        return label
