from snorkel.labeling import labeling_function
from wsee.preprocessors.preprocessors import *
from wsee.preprocessors.pattern_event_processor import parse_pattern_file, find_best_pattern_match, location_subtypes
from wsee.labeling import event_trigger_lfs

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


def parse_gaz_file(path):
    cause_consequence_mapping = {}
    with open(path, 'r') as gaz_reader:
        for line in gaz_reader.readlines():
            cols = line.split(' | ')
            cause_event = cols[0]
            consequence = re.findall('"([^"]*)"', cols[2])
            assert len(consequence) > 0
            cause_consequence_mapping[cause_event] = consequence[0]
    return cause_consequence_mapping


traffic_event_causes = parse_gaz_file('/Users/phuc/develop/python/wsee/data/traffic_event_causes.gaz')


# utility function
def check_required_args(entitiy_freqs):
    if any(loc_type in entitiy_freqs
           for loc_type in ['location', 'location_route', 'location_street', 'location_stop', 'location_city']) and \
            'trigger' in entitiy_freqs:
        return True
    else:
        return False


# location role
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_location_type(x):
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_route', 'location_city', 'location_stop']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
            return location
    return ABSTAIN


# delay role
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_delay_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['duration']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
                return delay
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_delay_event_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['duration']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and \
                    (event_trigger_lfs.lf_delay_cat(x) == event_trigger_lfs.Delay or
                     event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam):
                return delay
    return ABSTAIN


# direction role
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_argument_left_tokens, get_somajo_doc])
def lf_direction_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['location', 'location_city', 'location_stop']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and \
                    any(token.lower() in ['nach', 'richtung'] for token in x.argument_left_tokens[-1:]):
                return direction
    return ABSTAIN


# start_loc
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_start_location_type(x):
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_route', 'location_city', 'location_stop']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
            return start_loc
    return ABSTAIN


# end_loc
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_end_location_type(x):
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location', 'location_street', 'location_route', 'location_city', 'location_stop']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
            return end_loc
    return ABSTAIN


# start_date
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_start_date_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
                # TODO look at positional distance/preceding words to determine whether it is a start or a end date
                return start_date
    return ABSTAIN


# end_date
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_end_date_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['date', 'time']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN:
                # TODO look at positional distance/preceding words to determine whether it is a start or a end date
                return end_date
    return ABSTAIN


# cause
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_argument_left_tokens, get_somajo_doc])
def lf_cause_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['trigger']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and lf_not_an_event(x) == ABSTAIN and\
                    event_trigger_lfs.check_cause_keywords(x.argument_left_tokens[-4:]):
                return cause
    return ABSTAIN


@labeling_function(pre=[get_trigger_text, get_argument_text])
def lf_cause_gaz_file(x):
    if x.argument_text in traffic_event_causes:
        if traffic_event_causes[x.argument_text] == x.trigger_text:
            return cause
    else:
        return ABSTAIN


# jam_length
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_distance_type(x):
    if check_required_args(x.entity_type_freqs):
        arg_entity_type = x.argument['entity_type']
        if arg_entity_type in ['distance']:
            if lf_somajo_separate_sentence(x) == ABSTAIN and \
                    event_trigger_lfs.lf_trafficjam_cat(x) == event_trigger_lfs.TrafficJam:
                return jam_length
    return ABSTAIN


# route
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs,
                        get_argument, get_somajo_doc])
def lf_route_type(x):
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['location_route']:
        if lf_somajo_separate_sentence(x) == ABSTAIN and \
                event_trigger_lfs.lf_canceledstop_cat(x) == event_trigger_lfs.CanceledStop:
            return route
    return ABSTAIN


# no_args
@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs])
def lf_not_an_event(x):
    if event_trigger_lfs.lf_negative(x) == event_trigger_lfs.O:
        return no_arg
    else:
        return ABSTAIN


@labeling_function(pre=[get_trigger, get_argument, get_spacy_doc])
def lf_spacy_separate_sentence(x):
    # proof of concept that makes use of spaCy sentence splitter parsing feature
    # see: https://github.com/explosion/spaCy/blob/master/examples/information_extraction/entity_relations.py
    same_sentence = False
    for sentence in x.spacy_doc['sentences']:
        # edge case: two very similar sentences that both contain trigger text and arg text
        if x.spacy_doc['trigger'] in sentence and x.spacy_doc['argument'] in sentence:
            same_sentence = True
            break
    if same_sentence:
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[get_trigger, get_argument, get_stanford_doc])
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


@labeling_function(pre=[get_trigger, get_argument, get_somajo_doc])
def lf_somajo_separate_sentence(x):
    same_sentence = False

    for sentence in x.somajo_doc['sentences']:
        # edge case: two very similar sentences that both contain trigger text and arg text
        if x.somajo_doc['trigger'] in sentence and x.somajo_doc['argument'] in sentence:
            same_sentence = True
            break
    if same_sentence:
        return ABSTAIN
    else:
        return no_arg


# general
@labeling_function(pre=[get_trigger, get_argument, get_spacy_doc])
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


@labeling_function(resources=dict(rules=original_rules), pre=[get_trigger, get_trigger_idx,
                                                              get_argument, get_argument_idx,
                                                              get_mixed_ner])
def lf_event_patterns(x, rules):
    return event_patterns_helper(x, rules)


@labeling_function(resources=dict(rules=general_location_rules), pre=[get_trigger, get_trigger_idx,
                                                                      get_argument, get_argument_idx,
                                                                      get_mixed_ner])
def lf_event_patterns_general_location(x, rules):
    return event_patterns_helper(x, rules, general_location=True)
