# wsee - Weak Supervision for Event Extraction
Bachelor Thesis: "Investigating weak supervision for the extraction of mobility relations and events in German text"

## Set up
We recommend setting up a separate python virtual environment for this part of the code and then installing the requirements.
```shell script
pip install -r requirements.txt
```

We have created jupyter notebooks for all the experiments.
The [Data preparation](notebooks/data_preparation.ipynb) notebook guides you through downloading the corpus data, preparing it for the different experiments.
We recommend that you run all the commands in the [Data preparation](notebooks/data_preparation.ipynb) notebook before trying the other notebooks.

We used data provided by the German Research Center for Artificial Intelligence (DFKI).
The labeled data portion stems from the [DFKI SmartData Corpus](https://github.com/DFKI-NLP/smartdata-corpus) and is released as CC-BY 4.0. The accompanying paper:

_A German Corpus for Fine-Grained Named Entity Recognition and Relation Extraction of Traffic and Industry Events. Martin Schiersch, Veselina Mironova, Maximilian Schmitt, Philippe Thomas, Aleksandra Gabryszak, Leonhard Hennig. Proceedings of LREC, 2018._ [(bib)](paper.bib) [(pdf)](https://www.dfki.de/fileadmin/user_upload/import/9427_lrec_smartdata_corpus.pdf)

## Experiments
The following notebooks contain different experiments that we conducted as part of the labeling function development.
- [Sentence splitting experiments](notebooks/ssplit_experiments.ipynb): Here we compare different sentence splitters. We compare spaCy, stanfordnlp and somajo on selected examples and then compare the original sentence splitting with stanfordnlp in the corpus with somajo on the SD4M training data.
- [Trigger labeling experiments](notebooks/trigger_experiments.ipynb): Here we compare different strategies for trigger labeling functions.
- [Role labeling experiments](notebooks/role_experiments.ipynb): Here we compare different strategies for role labeling functions.

## Labeling
The following notebooks guide you through the trigger and role labeling process in the [pipeline](wsee/data/pipeline.py):
- [Trigger labeling](notebooks/event_type.ipynb): Shows the whole process of extracting event trigger candidates, performance of trigger labeling functions on the SD4M training data and labeling the Daystream data.
- [Role labeling](notebooks/event_arg_role.ipynb): Shows the whole process of extracting event role candidates, performance of role labeling functions on the SD4M training data and labeling the Daystream data.

The [Data preparation](notebooks/data_preparation.ipynb) notebook actually contains commands for creating the weakly labeled Daystream training dataset for event extraction.
Alternatively you can use the following command in the terminal of your choice:
```
python wsee/data/pipeline.py --input_path data/daystream_corpus --save_path data/daystream_corpus
```
You may need to adjust the input and save paths.