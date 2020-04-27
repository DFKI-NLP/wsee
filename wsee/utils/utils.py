import pickle
import json
import re
import pandas as pd
import numpy as np


def get_deep_copy(obj):
    return pickle.loads(pickle.dumps(obj))


def pretty_print_json(obj):
    print(json.dumps(json.loads(obj.to_json()), indent=2, ensure_ascii=False))


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


def let_most_probable_class_dominate(x: pd.Series) -> pd.Series:
    for event in x['event_triggers']:
        type_probs = np.asarray(event['event_type_probs'])
        max_idx = type_probs.argmax()
        type_probs *= 0.0
        type_probs[max_idx] = 1.0
        event['event_type_probs'] = list(type_probs)
    for role_pair in x['event_roles']:
        arg_probs = np.asarray(role_pair['event_argument_probs'])
        max_idx = arg_probs.argmax()
        arg_probs *= 0.0
        arg_probs[max_idx] = 1.0
        role_pair['event_argument_probs'] = list(arg_probs)
    return x
