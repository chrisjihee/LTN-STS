# LTN-STS

Semantic Textual Similarity (STS) is to measure the degree of semantic equivalence between two sentences. Our system do classification and regression for **[KLUE-STS](https://klue-benchmark.com/tasks/67/overview/description)** that is essential to other NLP tasks such as machine translation, summarization, and question answering. We implements the system using **[LTNtorch](https://github.com/bmxitalia/LTNtorch)**.

## Setup

After cloning this repository, make sure to install all the requirements.

- `git clone git@github.com:chrisjihee/LTN-STS.git`
- `pip3 install -r requirements.txt`

## Usage

After installation, please check the usage of the main module.

- `python3 main.py -h`
```
usage: main.py [-h] -t T [-n N] [-m M] [-k K] [-e E] [-lr LR] [-bs BS] [-msl MSL]

optional arguments:
  -h, --help  show this help message and exit
  -t T        task name: STS-CLS, STS-REG
  -n N        gpu id: 0, 1, 2, 3
  -m M        pretrained model id: 0, 1, 2, 3
  -k K        number of training samples
  -e E        number of training epochs
  -lr LR      learning rate
  -bs BS      batch size
  -msl MSL    max sequence length
```

## Run

After checking the usage, please run the main module with some proper options like following:

- `python3 main.py -t STS-CLS -n 0 -m 2 -k 100 -e 1`

## Results

Please check the results with following.

- [Classification](https://github.com/chrisjihee/LTN-STS/blob/master/expr2.ipynb): **0.8437(dev) F1** with KoELECTRA
- [Regression](https://github.com/chrisjihee/LTN-STS/blob/master/expr2.ipynb): **0.9290(dev) Pearson** with KoELECTRA

## Structure of repository

- `expr1.ipynb`: this notebook contains some experiments using LTN-STS with KoBERT.
- `expr2.ipynb`: this notebook contains some experiments using LTN-STS with KoELECTRA.
- `expr3.ipynb`: this notebook contains some experiments using LTN-STS with KoBigBird.
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
