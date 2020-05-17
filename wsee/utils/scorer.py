import copy
import pandas as pd
from typing import Tuple, Dict


class Result:
    def __init__(self, tp: int = 0, fp: int = 0, fn: int = 0):
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def precision(self):
        if self.tp + self.fp > 0:
            return self.tp/(self.tp + self.fp)
        else:
            return -1

    def recall(self):
        if self.tp + self.fn > 0:
            return self.tp/(self.tp + self.fn)
        else:
            return -1

    def f1(self):
        p = self.precision()
        r = self.recall()
        if p + r > 0:
            return 2 * (p*r)/(p+r)
        else:
            return -1


def argument_equals(pred_arg, gold_arg):
    if pred_arg['role'] != gold_arg['role']:
        return False
    if pred_arg == gold_arg:
        return True
    if pred_arg['id'] == gold_arg['id']:
        return True
    if pred_arg['start'] != gold_arg['start'] or pred_arg['end'] != gold_arg['end']:
        return False
    return pred_arg['entity_type'] == gold_arg['entity_type']


def get_event_span(event):
    start = [event['trigger']['start']]
    end = [event['trigger']['end']]
    for arg in event['arguments']:
        arg_start = arg['start']
        arg_end = arg['end']
        if arg_start < start:
            start = arg_start
        if arg_end > end:
            end = arg_end
    return start, end


def event_equals(pred_event, gold_event, ignore_span=False, ignore_args=False):
    if pred_event == gold_event:
        return True
    if pred_event['event_type'] != gold_event['event_type']:
        return False
    if not ignore_span:
        pred_event_start, pred_event_end = get_event_span(pred_event)
        gold_event_start, gold_event_end = get_event_span(gold_event)
        if pred_event_start != gold_event_start or pred_event_end != gold_event_end:
            return False
    if not ignore_args:
        if len(pred_event['arguments']) != len(gold_event['arguments']):
            return False
        for gold_arg in gold_event['arguments']:
            found_arg = False
            # TODO Add option to only check required arg, i.e. location
            if any(argument_equals(gold_arg, pred_arg) for pred_arg in pred_event['arguments']):
                found_arg = True
            if not found_arg:
                return False
    return True


def event_subsumes(subsumed_event, subsuming_event, ignore_span=False, ignore_args=False):
    if subsumed_event == subsuming_event:
        return True
    if subsumed_event['event_type'] != subsuming_event['event_type']:
        return False
    if not ignore_span:
        subsumed_event_start, subsumed_event_end = get_event_span(subsumed_event)
        subsuming_event_start, subsuming_event_end = get_event_span(subsuming_event)
        if subsumed_event_start < subsuming_event_start or subsumed_event_end > subsuming_event_end:
            return False
    if not ignore_args:
        if len(subsumed_event['arguments']) > len(subsuming_event['arguments']):
            return False
        for subsumed_arg in subsumed_event['arguments']:
            found_arg = False
            # TODO Add option to only check required arg, i.e. location
            if any(argument_equals(subsuming_arg, subsumed_arg) for subsuming_arg in subsuming_event['arguments']):
                found_arg = True
            if not found_arg:
                return False
    return True


def event_scorer(pred_events, gold_events, allow_subsumption=False) -> Tuple[int, int, int]:
    pred_events_copy = copy.deepcopy(pred_events) if pred_events else []
    gold_events_copy = copy.deepcopy(gold_events) if gold_events else []
    # sort event lists by event span start
    pred_events_copy.sort(key=lambda event: get_event_span(event)[0])
    gold_events_copy.sort(key=lambda event: get_event_span(event)[0])
    tp = 0
    fn = 0

    for gold_event in gold_events_copy:
        found_idx = -1
        for idx, pred_event in enumerate(pred_events_copy):
            if event_equals(pred_event, gold_event) or \
                    (allow_subsumption and
                     (event_subsumes(subsumed_event=pred_event, subsuming_event=gold_event) or
                      event_subsumes(subsumed_event=gold_event, subsuming_event=pred_event))):
                tp += 1
                found_idx = idx
        if found_idx < 0:
            fn += 1
        else:
            del pred_events_copy[found_idx]
    fp = len(pred_events_copy)
    return tp, fp, fn


def event_by_class_scorer(pred_events, gold_events, allow_subsumption=False) -> Dict[str, Result]:
    results: Dict[str, Result] = {}
    event_types = set([event['event_type'] for event in pred_events])
    event_types += set([event['event_type'] for event in gold_events])
    for event_type in event_types:
        class_pred_events = [event for event in pred_events if event['event_type'] == event_type]
        class_gold_events = [event for event in gold_events if event['event_type'] == event_type]
        result: Tuple[int, int, int] = event_scorer(class_pred_events, class_gold_events, allow_subsumption)
        results[event_type] = Result(*result)
    return results


def score_document(pred_doc, gold_doc, allow_subsumption=False):
    results = event_by_class_scorer(pred_doc['events'], gold_doc['events'], allow_subsumption)
    return results


def score_files(pred_file_path, gold_file_path, allow_subsumption=False):
    pred_file = pd.read_json(pred_file_path, lines=True, encoding='utf8')
    gold_file = pd.read_json(gold_file_path, lines=True, encoding='utf8')

    # TODO: Tuple does not support item assignment, write result class
    results: Dict[str, Result] = {}
    # TODO probably not a good idea to do it all in memory
    for pred_doc_events, gold_doc_events in zip(list(pred_file['events']), list(gold_file['events'])):
        doc_results = event_by_class_scorer(pred_doc_events, gold_doc_events, allow_subsumption)
        for event_type, result in doc_results:
            if event_type in results:
                results[event_type].tp += result.tp
                results[event_type].fp += result.fp
                results[event_type].fn += result.fn
            else:
                results[event_type] = result
    for event_type, result in results:
        formatted_results = '\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(result.precision(), result.recall(), result.f1())
        print(f'{event_type}: {formatted_results}')
