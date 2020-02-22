import argparse
import json
from pathlib import Path

from fastavro import reader
from tqdm import tqdm

SD4M_RELATION_TYPES = ['Accident', 'CanceledRoute', 'CanceledStop', 'Delay',
                       'Obstruction', 'RailReplacementService', 'TrafficJam']


def main():
    parser = argparse.ArgumentParser(
        description='Smartdata avro schema to jsonl converter')
    parser.add_argument('input', help='Directory of avro file structure')
    parser.add_argument('-f', dest='force_output', action='store_true',
                        help='Force creation of new output folder')
    args = parser.parse_args()

    input_path = Path(args.input)
    assert input_path.exists(), 'Input not found: %s'.format(args.input)

    for split in ['train', 'dev', 'test']:
        input_file = input_path.joinpath(split, f'{split}.avro')
        output_file = input_path.joinpath(split, f'{split}_with_events.jsonl')
        assert not output_file.exists() or args.force_output, 'Output already exists'
        print('Writing to', output_file)
        convert_file(input_file, output_file)
    print('Done.')


def convert_file(avro_file_path, jsonl_file_path):
    with avro_file_path.open('rb') as avro_file, jsonl_file_path.open('w', encoding='utf-8') as jsonl_file:
        avro_reader = reader(avro_file)
        for doc in tqdm(avro_reader):
            is_smart_data_doc = 'fileName' in doc['refids'] and doc['refids']['fileName'].startswith('smartdata')
            if is_smart_data_doc:
                json_line = convert_document(doc)
                jsonl_file.write(json_line + '\n')


def span_to_token_idxs(tokens, span):
    start_idx = next(idx
                     for idx, token in enumerate(tokens)
                     if token['span']['start'] == span['start'])
    end_idx = next(idx
                   for idx, token in enumerate(tokens)
                   if token['span']['end'] == span['end'])
    return start_idx, end_idx


def convert_document(doc):
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
            rm for rm in doc['relationMentions'] if rm['name'] in SD4M_RELATION_TYPES
        ]
        for relation in filtered_relations:
            # There are a few events with more than a single trigger
            trigger = [arg for arg in relation['args'] if arg['role'] == 'trigger'][0]
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
    jsonl_ner_tags = [t['ner'] for t in doc['tokens']]

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
        return {
            'id': entity['id'],
            'text': span_to_text(text, entity['span']),
            'entity_type': entity['type'],
            'start': start_token_idx,
            'end': end_token_idx + 1  # Generate exclusive token spans
        }
    except StopIteration:
        print('Token offset issue')
        return None


def span_to_text(text, span):
    return text[span['start']:span['end']]


if __name__ == '__main__':
    main()

