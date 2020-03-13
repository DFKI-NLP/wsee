import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from wsee.utils import utils


def load_data(path, use_build_defaults=True):
    input_path = Path(path)
    assert input_path.exists(), 'Input not found: %s'.format(path)

    output_dict = {}
    for split in ['train', 'dev', 'test']:
        if use_build_defaults:
            sd_path = input_path.joinpath(split, f'{split}_with_events_and_defaults.jsonl')
        else:
            sd_path = input_path.joinpath(split, f'{split}_with_events.jsonl')
        assert os.path.exists(sd_path)
        sd_data = pd.read_json(sd_path, lines=True)
        output_dict[split] = sd_data

    daystream_path = os.path.join(input_path, 'daystream.jsonl')
    assert os.path.exists(daystream_path)
    daystream = pd.read_json(daystream_path, lines=True)
    output_dict['daystream'] = daystream

    return output_dict


def build_event_trigger_examples(dataframe):
    event_type_rows = []
    event_type_rows_y = []

    event_count = 0

    print(f"DataFrame has {len(dataframe.index)} rows")
    for index, row in tqdm(dataframe.iterrows()):
        for event_trigger in row.event_triggers:
            augmented_row = utils.get_deep_copy(row)
            augmented_row['trigger_id'] = event_trigger['id']
            event_type_rows.append(augmented_row)
            event_type_num = np.asarray(event_trigger['event_type_probs']).argmax()
            event_type_rows_y.append(event_type_num)
            if event_type_num != 7:
                event_count += 1

    print("Number of events:", event_count)
    event_type_rows = pd.DataFrame(event_type_rows)
    event_type_rows_y = np.asarray(event_type_rows_y)
    return event_type_rows, event_type_rows_y


def build_event_roles_examples(dataframe):
    event_role_rows_list = []
    event_role_rows_y = []

    event_count = 0

    for index, row in tqdm(dataframe.iterrows()):
        for event_role in row.event_roles:
            augmented_row = utils.get_deep_copy(row)
            augmented_row['trigger_id'] = event_role['trigger']
            augmented_row['argument_id'] = event_role['argument']
            event_role_rows_list.append(augmented_row)
            event_role_num = np.asarray(event_role['event_argument_probs']).argmax()
            event_role_rows_y.append(event_role_num)
            if event_role_num != 10:
                event_count += 1

    print("Number of event roles:", event_count)
    event_role_rows = pd.DataFrame(event_role_rows_list).reset_index(drop=True)
    event_role_rows_y = np.asarray(event_role_rows_y)

    return event_role_rows, event_role_rows_y
