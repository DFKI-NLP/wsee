from snorkel.labeling import labeling_function
from wsee.preprocessors.preprocessors import *
from wsee.preprocessors.pattern_event_processor import parse_pattern_file, find_best_pattern_match, location_subtypes


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


@labeling_function(pre=[get_trigger, get_argument, get_between_distance, get_trigger_left_tokens])
def lf_date_type(x):
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['DATE', 'TIME', 'date', 'time']:
        if x.between_distance < 5:
            # TODO look at positional distance/preceding words to determine whether it is a start or a end date
            return start_date
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_argument, get_between_distance])
def lf_location_type(x):
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['LOCATION', 'LOCATION_STREET', 'LOCATION_ROUTE', 'LOCATION_CITY', 'LOCATION_STOP']:
        if x.between_distance < 5:
            return location
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_argument, get_between_distance])
def lf_distance_type(x):
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['distance', 'DISTANCE']:
        if x.between_distance < 5:
            return jam_length
    return ABSTAIN


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


@labeling_function(pre=[get_trigger, get_argument, get_spacy_doc])
def lf_spacy_separate_sentence(x):
    # proof of concept that makes use of spaCy sentence splitter parsing feature
    # see: https://github.com/explosion/spaCy/blob/master/examples/information_extraction/entity_relations.py
    same_sentence = False
    for sentence in x.spacy_doc.sents:
        if x.trigger['text'] in sentence.text and x.argument['text'] in sentence.text:
            same_sentence = True
            break
    if same_sentence:
        return ABSTAIN
    else:
        return no_arg


@labeling_function(pre=[get_trigger, get_argument, get_stanford_doc])
def lf_stanford_separate_sentence(x):
    # proof of concept that makes use of spaCy sentence splitter parsing feature
    # see: https://github.com/explosion/spaCy/blob/master/examples/information_extraction/entity_relations.py
    same_sentence = False

    sentences = [" ".join([t.text for t in s.tokens]) for s in x.stanford_doc.sentences]

    for sentence in sentences:
        if x.trigger['text'] in sentence and x.argument['text'] in sentence:
            same_sentence = True
            break
    if same_sentence:
        return ABSTAIN
    else:
        return no_arg


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
