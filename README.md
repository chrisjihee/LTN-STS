# LTN-STS

Semantic Textual Similarity (STS) is to measure the degree of semantic equivalence between two sentences. We do experiments on [KLUE-STS](https://klue-benchmark.com/tasks/67/overview/description) that is essential to other NLP tasks such as machine translation, summarization, and question answering.

We implements a system using [LTNtorch](https://github.com/bmxitalia/LTNtorch). Our system do binary classification task (and regression task).

# Installation

If you install LTN-STS by cloning this repository, make sure to install all the requirements.

`pip3 install -r requirements.txt`

# Structure of repository

- `main.py`: this module contains the implementation of LTN-STS.
- `data.py`: this module contains the implementation of converting original KLUE-STS dataset to each task specific dataset.
- `data/`: this folder contains the data for our experiments.
- `examples/`: this folder contains some problems approached using LTN.
- `tutorials/`: this folder contains some important tutorials to getting started with coding in LTN.

# Acknowledgements

LTN-STS has been developed thanks to the following people.

- Seung-Hoon Na (JBNU)
- Tommaso Carraro (FBK)
- Samy Badreddine (Sony AI)
