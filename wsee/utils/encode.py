import numpy as np


def one_hot_encode(label, label_names, negative_label=None):
    if negative_label is None:
        negative_label = label_names[-1]
    label = label if label in label_names else negative_label
    class_probs = [1.0 if label_name == label else 0.0 for label_name in label_names]
    return class_probs


def one_hot_decode(class_probs, label_names):
    class_probs_array = np.asarray(class_probs)
    class_name = label_names[class_probs_array.argmax()]
    return class_name
