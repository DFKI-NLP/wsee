from snorkel.labeling import labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function
from fuzzywuzzy import process
from wsee.preprocessors.preprocessors import get_trigger, get_argument, get_left_tokens, get_right_tokens, \
    get_entity_type_freqs, get_between_distance


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


@labeling_function(pre=[get_trigger, get_argument, get_between_distance])
def lf_date_type(x):
    # purely distance based for now: could use dependency parsing/ context words
    arg_entity_type = x.argument['entity_type']
    if arg_entity_type in ['DATE', 'TIME', 'date', 'time']:
        if x.between_distance < 10:
            # TODO look at positional distance/preceding words to determine whether it is a start or a end date
            return start_date
    return ABSTAIN
