## fence-detection
Research internship with the Gemeente Amsterdam concerning the detection of fencing along open water within the city center.

![img.png](media/img.png)

[comment]: <> (![]&#40;media/examples/emojis.png&#41;)

---


## Project Folder Structure

There are the following folders in the structure:

1) [`data`](./data): Placeholder that should contain the annotated dataset and geometry
1) [`experiments`](./experiments): Placeholder for train- and validation-logs and model weights
1) [`loaders`](./loaders): Folder containing the panorama- and dataloaders for training
1) [`models`](./models): Folder containing the models and training code
1) [`notebooks`](./notebooks): Folder containing Jupyter Notebooks for visualizations
1) [`scripts`](./notebooks): Folder containing scripts for inference, annotation converters, and data splits
1) [`utils`](./notebooks): Folder containing augmentation, logging, metrics, and other functions

---


## Installation

In order to run all steps of the lexical simplification pipeline, follow these steps:

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/fence-detection
    ```
1) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---


## Usage

<!-- Simplifications can be made on English and Dutch. They require a number of files:

**Deploying the pipeline for English** 
1) Download a word embedding model from (fasttext) and store it in the models folder as __crawl-300d-2M-subword.vec__
1) Download the BenchLS, NNSeval and lex.mturk datasets from https://simpatico-project.com/?page_id=109 dataset and store them in the models folder

**Deploying the pipeline for Dutch**
1) Download the word embedding model from https://dumps.wikimedia.org/nlwiki/20160501/ and store it in the models folder as __wikipedia-320.txt__


Then the model can be run as follows:
```
python3 BERT_for_LS.py --model GroNLP/bert-base-dutch-cased --eval_dir ../datasets/Dutch/dutch_data.txt
```

|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--model`| str| `the name of the model that is used for generating the predictions: a path to a folder or a huggingface directory.`|  -|
|`--eval_dir`| str| `path to the file with the to-be-simplified sentences.`| -|
|`--results_file`|  str | `path to file where the performance report is written out`| -|
|`--analysis`| Bool| `whether or not to output all the generated candidates and the reason for their removal `|False|
|`--ranking`| Bool| `whether or not to perform ranking of the generated candidates`|False|
|`--evaluation`| Bool| `whether or not to perform an evaluation of the generated candidates`|True|

|---|:---:|:---:|:---:|

--- -->

## Finetuning a Model

<!-- Can be done in one of two ways: todo explain -->

## How it works


## Acknowledgements

Our segmentation models use [Segmentation Models for PyTorch](https://github.com/qubvel/segmentation_models.pytorch) by [qubvel](https://github.com/qubvel/): 