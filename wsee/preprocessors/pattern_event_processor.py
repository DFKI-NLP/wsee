import re
from typing import List


escaped_chars = ["<", "(", "[", "{", "\\", "^", "-", "=", "$", "!", "|",
                 "]", "}", ")", "?", "*", "+", ".", ">"]

location_subtypes = ['LOCATION', 'LOCATION_STREET', 'LOCATION_ROUTE', 'LOCATION_CITY', 'LOCATION_STOP']

ner_mapping = {
    'LOCATION-STREET': 'LOCATION_STREET',
    'LOCATION-ROUTE': 'LOCATION_ROUTE',
    'LOCATION-CITY': 'LOCATION_CITY',
    'LOCATION-STOP': 'LOCATION_STOP',
}


def fix_ner_tags(sequence, mapping):
    for key in mapping.keys():
        sequence = re.sub(key, mapping[key], sequence)
    return sequence


class ConverterRuleSlot:
    def __init__(self, position, entity_type, role):
        self.position = position
        self.entity_type = entity_type
        self.role = role

    def __repr__(self):
        return f"ConverterRuleSlot(position: {self.position}, entity_type: {self.entity_type}, role: {self.role})"


class ConverterRule:
    def __init__(self):
        self.pattern = None
        self.rule_id = None
        # what does relation type do?
        self.relation_type = None
        self.slots: List[ConverterRuleSlot] = []

    def __repr__(self):
        return f"ConverterRule(pattern: {self.pattern}, slots: {self.slots})"


def parse_pattern_file(pattern_file):
    tag_pattern = re.compile('\\[.*?]')
    rules = {}
    with open(pattern_file, 'r') as pattern_reader:
        for idx, line in enumerate(pattern_reader.readlines()):
            rule: ConverterRule = ConverterRule()
            relation_type = None

            line = line.strip()
            if len(line) == 0:
                continue
            pattern = ""
            matches = tag_pattern.finditer(line)
            offset = 0
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                annotation = match.group()[1:-1]

                tmp = line[:start_pos]
                index = tmp.rfind(' ')
                if index < 0:
                    index = 0
                else:
                    index += 1
                # ignore hashtags for type
                if tmp.rfind('#') == index:
                    index += 1

                entity_type = tmp[index:]

                # add mapping for slots
                position = line[:start_pos].count(entity_type)
                for part in annotation.split(','):
                    slot: ConverterRuleSlot = ConverterRuleSlot(position, entity_type, part)
                    if part.lower() == 'relationtype':
                        relation_type = slot
                    else:
                        rule.slots.append(slot)

                pattern += line[offset:start_pos]
                offset = end_pos
            pattern += line[offset:]
            pattern = fix_ner_tags(pattern, ner_mapping)

            rule.relation_type = relation_type
            rule.id = idx
            rule.pattern = convert_to_regex(pattern)

            rules[rule.pattern] = rule

    return rules


def escape_regex_chars(pattern, optional_hashtag=True):
    escaped_pattern = ''
    for idx, char in enumerate(pattern):
        if char in escaped_chars:
            escaped_pattern += '\\'
        escaped_pattern += char
        if optional_hashtag and char == '#':
            escaped_pattern += '?'

    return escaped_pattern


def add_location_subtypes(pattern):
    loc_pattern = re.compile('LOCATION(?!_)')
    return loc_pattern.sub("(" + '|'.join(location_subtypes) + ")", pattern)


def convert_to_regex(pattern):
    converted_rule_pattern = escape_regex_chars(pattern)
    converted_rule_pattern = add_location_subtypes(converted_rule_pattern)
    return re.compile(converted_rule_pattern)


def find_best_pattern_match(pattern, rules, trigger_spans, argument_spans):
    # pattern has to be mixed ner pattern from text containing the relevant trigger/trigger-arg pair
    best_match = None
    best_rule = None
    for p in rules.keys():
        matches = p.finditer(pattern)
        for match in matches:
            if match.start() <= min(trigger_spans[0], argument_spans[0]) \
                    and match.end() >= max(trigger_spans[1], argument_spans[1]):
                if len(match.group()) > len(best_match.group()):
                    best_match = match
                    best_rule = rules[p]
    return best_rule, best_match
