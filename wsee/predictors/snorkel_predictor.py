import pandas as pd
from pathlib import Path
from typing import Union, Tuple

from snorkel.labeling import LabelModel, PandasLFApplier
from wsee.data import pipeline, ace_formatter
from wsee.utils import utils


def load_snorkel_ee_components(save_path: Union[str, Path]) \
        -> Tuple[LabelModel, LabelModel]:
    save_path = Path(save_path)
    assert save_path.exists(), f"Save path does not exist: {save_path}"

    trigger_label_model: LabelModel = LabelModel()
    trigger_label_model.load(Path(save_path).joinpath('trigger_lm.pt'))
    role_label_model: LabelModel = LabelModel()
    role_label_model.load(Path(save_path).joinpath('role_lm.pt'))

    return trigger_label_model, role_label_model


def predict_documents(documents: pd.DataFrame, trigger_label_model: LabelModel,
                      role_label_model: LabelModel):
    if 'event_triggers' not in documents and 'event_roles' not in documents:
        documents = documents.apply(pipeline.add_default_events, axis=1)

    # 1. Get trigger probabilities
    df_predict_triggers, _ = pipeline.build_event_trigger_examples(documents)
    trigger_lf_applier = PandasLFApplier(pipeline.get_trigger_list_lfs())
    L_predict_triggers = trigger_lf_applier.apply(df_predict_triggers)
    event_trigger_probs = trigger_label_model.predict_proba(L_predict_triggers)

    merged_event_trigger_examples = pipeline.merge_event_trigger_examples(
        df_predict_triggers, utils.zero_out_abstains(event_trigger_probs, L_predict_triggers))

    # 2. Get role probabilities
    df_predict_roles, _ = pipeline.build_event_role_examples(documents)
    role_lf_applier = PandasLFApplier(pipeline.get_role_list_lfs())
    L_predict_roles = role_lf_applier.apply(df_predict_roles)
    event_roles_probs = role_label_model.predict_proba(L_predict_roles)

    merged_event_role_examples = pipeline.merge_event_role_examples(
        df_predict_roles, utils.zero_out_abstains(event_roles_probs, L_predict_roles))

    # 3. Update documents with trigger & role probabilities
    labeled_documents: pd.DataFrame = documents.copy()
    # Make sure to remove event_triggers and roles that were built per default
    for idx, row in labeled_documents.iterrows():
        row['event_triggers'] = []
        row['event_roles'] = []
    if 'id' in labeled_documents:
        labeled_documents.set_index('id', inplace=True)

    triggers = merged_event_trigger_examples[['event_triggers']]
    roles = merged_event_role_examples[['event_roles']]

    labeled_documents.update(triggers)
    labeled_documents.update(roles)

    labeled_documents.reset_index(level=0, inplace=True)

    # 4. Add ACE events
    labeled_documents = ace_formatter.snorkel_to_ace_format(labeled_documents)
    return labeled_documents
