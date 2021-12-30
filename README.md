# LTN-STS

Semantic Textual Similarity (STS) is to measure the degree of semantic equivalence between two sentences. Our system do classification and regression for **[KLUE-STS](https://klue-benchmark.com/tasks/67/overview/description)** that is essential to other NLP tasks such as machine translation, summarization, and question answering. We implements the system using **[LTNtorch](https://github.com/bmxitalia/LTNtorch)**.

## Setup

After cloning this repository, make sure to install all the requirements.

- `git clone git@github.com:chrisjihee/LTN-STS.git`
- `pip3 install -r requirements.txt`

## Run

After installation, please run the main module.

- `cd LTN-STS`
- `python3 main.py`

## Results

Please check the results with following.

- [Classification](https://github.com/chrisjihee/LTN-STS/blob/master/expr.ipynb): **0.8320(dev) F1** with KoELECTRA-base
- Regression: to be reported

## Structure of repository

- `expr.ipynb`: this notebook contains some experiments using LTN-STS and their results.
- `main.py`: this module contains the implementation of LTN-STS.
- `data.py`: this module contains converting original KLUE-STS dataset to each task-specific dataset.
- `data/`: this folder contains the data for our experiments.

## Acknowledgements

LTN-STS has been developed thanks to the following people.

- [Tommaso Carraro](https://github.com/bmxitalia)
- [Samy Badreddine](https://www.ai.sony/people/c6ecb9ab786d5b75047f5b00515dc67bae284640)
- [Seung-Hoon Na](https://nlp.jbnu.ac.kr/~nash/faculty.html)
- [Sungjoon Park](https://sungjoonpark.github.io)
- [Jangwon Park](https://github.com/monologg)
