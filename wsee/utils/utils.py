import pickle
import json
import re


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
