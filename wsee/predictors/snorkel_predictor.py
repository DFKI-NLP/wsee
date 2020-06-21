import pickle
import pandas as pd
from pathlib import Path
from typing import Union, Tuple

from snorkel.labeling import LabelModel, PandasLFApplier, filter_unlabeled_dataframe
from wsee.data import pipeline, ace_formatter


def load_snorkel_ee_components(save_path: Union[str, Path]) \
        -> Tuple[PandasLFApplier, LabelModel, PandasLFApplier, LabelModel]:
    save_path = Path(save_path)
    assert save_path.exists(), f"Save path does not exist: {save_path}"
    with open(save_path, 'rb') as pickled_file:
        snorkel_ee_components = pickle.load(pickled_file)
        trigger_lf_applier: PandasLFApplier = snorkel_ee_components["trigger_lf_applier"]
        trigger_label_model: LabelModel = snorkel_ee_components["trigger_label_model"]
        role_lf_applier: PandasLFApplier = snorkel_ee_components["role_lf_applier"]
        role_label_model: LabelModel = snorkel_ee_components["role_label_model"]
        return trigger_lf_applier, trigger_label_model, role_lf_applier, role_label_model


def predict_documents(documents: pd.DataFrame, trigger_lf_applier: PandasLFApplier, trigger_label_model: LabelModel,
                      role_lf_applier: PandasLFApplier, role_label_model: LabelModel):
    # 1. Get trigger probabilities
    df_predict_triggers, _ = pipeline.build_event_trigger_examples(documents)
    L_predict_triggers = trigger_lf_applier.apply(df_predict_triggers)
    event_trigger_probs = trigger_label_model.predict_proba(L_predict_triggers)
    df_predict_triggers_filtered, event_trigger_probs_filtered = filter_unlabeled_dataframe(
        X=df_predict_triggers, y=event_trigger_probs, L=L_predict_triggers
    )
    merged_event_trigger_examples = pipeline.merge_event_trigger_examples(
        df_predict_triggers_filtered, event_trigger_probs_filtered)

    # 2. Get role probabilities
    df_predict_roles, _ = pipeline.build_event_role_examples(documents)
    L_predict_roles = role_lf_applier.apply(df_predict_roles)
    event_roles_probs = role_label_model.predict_proba(L_predict_roles)
    df_predict_roles_filtered, event_roles_probs_filtered = filter_unlabeled_dataframe(
        X=df_predict_roles, y=event_roles_probs, L=L_predict_roles
    )
    merged_event_role_examples = pipeline.merge_event_role_examples(
        df_predict_roles_filtered, event_roles_probs_filtered)

    # 3. Update documents with trigger & role probabilities
    labeled_documents: pd.DataFrame = documents.copy()
    # Make sure to remove event_triggers and roles that were built per default during the avro-json conversion
    for idx, row in labeled_documents.iterrows():
        row['event_triggers'] = []
        row['event_roles'] = []
    if 'id' in labeled_documents:
        labeled_documents.set_index('id', inplace=True)

    labeled_documents.update(
        merged_event_trigger_examples.drop(['text', 'tokens', 'ner_tags', 'entities'], axis=1, inplace=True))
    labeled_documents.update(
        merged_event_role_examples.drop(['text', 'tokens', 'ner_tags', 'entities'], axis=1, inplace=True))

    labeled_documents.reset_index(level=0, inplace=True)

    # 4. Add ACE events
    labeled_documents = ace_formatter.snorkel_to_ace_format(labeled_documents)
    return labeled_documents
