"""
Converts avro files with n-ary relations into jsonl files with events. Each event can have
multiple arguments but must include at least one argument with the trigger role. Some relations are
dropped for which is is not the case. The disaster relation includes a 'disaster_type' argument
which is identical to a trigger and is thus converted into the trigger type at each occurrence.
"""
import argparse
import json
from pathlib import Path

from fastavro import reader
from tqdm import tqdm

SD4M_RELATION_TYPES = ['Accident', 'CanceledRoute', 'CanceledStop', 'Delay',
                       'Obstruction', 'RailReplacementService', 'TrafficJam']

# Removed SDW relations without a trigger type:
# ["OrganizationLeadership", "Acquisition"]
SDW_RELATION_TYPES = ["Disaster", "Insolvency", "Layoffs", "Merger", "SpinOff", "Strike",
                      "CompanyProvidesProduct", "CompanyUsesProduct", "CompanyTurnover",
                      "CompanyRelationship", "CompanyFacility", "CompanyIndustry",
                      "CompanyHeadquarters", "CompanyWebsite", "CompanyWikipediaSite",
                      "CompanyNumEmployees", "CompanyCustomer", "CompanyProject",
                      "CompanyFoundation", "CompanyTermination", "CompanyFinancialEvent"]


def main():
    parser = argparse.ArgumentParser(
        description='Smartdata avro schema to jsonl converter')
    parser.add_argument('input', help='Directory of avro file structure')
    parser.add_argument('-f', dest='force_output', action='store_true',
                        help='Force creation of new output folder')
    args = parser.parse_args()

    input_path = Path(args.input)
    assert input_path.exists(), 'Input not found: %s'.format(args.input)

    convert_avros(input_path, args.force_output)


def convert_avros(input_path, force_output=None):
    dataset_types = [
        ('sdw', SDW_RELATION_TYPES),
        ('sd4m', SD4M_RELATION_TYPES),
    ]

    for suffix, relations in dataset_types:
        for split in ['train', 'dev', 'test']:
            input_file = input_path.joinpath(split, f'{split}.avro')
            if not input_file.exists():
                input_file = input_path.joinpath(split, '1.avro')
            assert input_file.exists(), 'Input file not found'
            output_file = input_path.joinpath(split, f'{split}_{suffix}_with_events.jsonl')
            assert not output_file.exists() or force_output, 'Output already exists'
            print('Writing to', output_file)
            convert_file(input_file, output_file, relations)
    print('Done.')


def convert_file(avro_file_path, jsonl_file_path, relations):
    with avro_file_path.open('rb') as avro_file, jsonl_file_path.open('w') as jsonl_file:
        avro_reader = reader(avro_file)
        for doc in tqdm(avro_reader):
            if _is_smart_data_doc(doc):
                json_line = convert_document(doc, relations)
                jsonl_file.write(json_line + '\n')


def span_to_token_idxs(tokens, span):
    start_idx = next(idx
                     for idx, token in enumerate(tokens)
                     if token['span']['start'] == span['start'])
    end_idx = next(idx
                   for idx, token in enumerate(tokens)
                   if token['span']['end'] == span['end'])
    return start_idx, end_idx


def convert_document(doc, relations):
    jsonl_entities = []
    for entity in doc['conceptMentions']:
        converted_entity = convert_entity(text=doc['text'],
                                          tokens=doc['tokens'],
                                          entity=entity)
        if converted_entity:
            jsonl_entities.append(converted_entity)

    jsonl_events = []
    if doc['relationMentions']:
        filtered_relations = [
            rm for rm in doc['relationMentions'] if rm['name'] in relations
        ]
        for relation in filtered_relations:
            if relation['name'] == 'Disaster':
                disaster_type_args = [a for a in relation['args'] if a['role'] == 'type']
                assert len(disaster_type_args) == 1
                disaster_type_args[0]['role'] = 'trigger'
            # There are a few events with more than a single trigger
            triggers = [arg for arg in relation['args'] if arg['role'] == 'trigger']
            if len(triggers) == 0:
                continue
            # Consistency: If "fallen"/"fällt" and "aus" in triggers choose the former as trigger
            if len(triggers) > 1:
                aus_trigger = next(
                    (trigger for trigger in triggers if
                     trigger['conceptMention']['normalizedValue'] in ['aus', 'aus.']), None)
                fallen_trigger = next(
                    (trigger for trigger in triggers
                     if trigger['conceptMention']['normalizedValue'] in ['fällt', 'faellt',
                                                                         'fallen']), None)
                if aus_trigger and fallen_trigger:
                    trigger = fallen_trigger
                else:
                    trigger = triggers[0]
            else:
                trigger = triggers[0]
            jsonl_trigger = convert_entity(doc['text'], doc['tokens'], trigger['conceptMention'])
            if jsonl_trigger is None:
                print('Skipping invalid event')
                continue
            args = [arg for arg in relation['args'] if arg['role'] != 'trigger']
            jsonl_args = []
            for arg in args:
                jsonl_arg = convert_entity(doc['text'], doc['tokens'], arg['conceptMention'])
                if jsonl_arg:
                    jsonl_arg['role'] = arg['role']
                    jsonl_args.append(jsonl_arg)
            if len(jsonl_args) == 0:
                print('Skipping invalid event')
                continue
            jsonl_events.append({
                'event_type': relation['name'],
                'trigger': jsonl_trigger,
                'arguments': jsonl_args
            })

    text_tokens = [span_to_text(doc['text'], t['span']) for t in doc['tokens']]
    jsonl_ner_tags = _convert_ner_tags([t['ner'] for t in doc['tokens']])

    return json.dumps({
        'id': doc['id'],
        'text': doc['text'],
        'tokens': text_tokens,
        'ner_tags': jsonl_ner_tags,
        'entities': jsonl_entities,
        'events': jsonl_events
    }, ensure_ascii=False)


def convert_entity(text, tokens, entity):
    try:
        start_token_idx, end_token_idx = span_to_token_idxs(tokens, entity['span'])
        # The disaster type relations are not annotated with a trigger but with a disaster type.
        # However they can be used interchangeably, thus convert disaster types to triggers.
        entity_type = entity['type'].lower()
        if entity_type == 'disaster-type':
            entity_type = 'trigger'
        elif entity_type == 'disaster_type':
            entity_type = 'trigger'
        return {
            'id': entity['id'],
            'text': span_to_text(text, entity['span']),
            'entity_type': entity_type,
            'start': start_token_idx,
            'end': end_token_idx + 1  # Generate exclusive token spans
        }
    except StopIteration:
        print('Token offset issue')
        return None


def _convert_ner_tags(ner_tags):
    return [ner_tag[:2] + 'TRIGGER' if ner_tag != 'O' and ner_tag[2:] == 'DISASTER_TYPE'
            else ner_tag
            for ner_tag in ner_tags]


def span_to_text(text, span):
    return text[span['start']:span['end']]


def _is_smart_data_doc(doc):
    return 'fileName' in doc['refids'] and doc['refids']['fileName'].startswith('smartdata')


if __name__ == '__main__':
    main()
