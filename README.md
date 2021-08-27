# Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis

This repository contains code for our EMNLP 2020 paper: 

Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis


## Datasets

The training data comes from four domains: Restaurant(R) 、 Laptop(L) 、 Service(S) 、 Devices(D).  
For each domain transfer pairs, the unlabeled data come from a combination of training data from the two domains(ratio: 1:1).

The in-domain corpus(used for training BERT-E) come from [yelp](https://www.yelp.com/dataset/challenge) and [amazon reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). 

Click here to get [BERT-E](https://pan.baidu.com/s/1hNyNCyfOHzznuPbxT1LNFQ)(BERT-Extented),and  the extraction code is by0i.(Please specify the directory where BERT is stored in modelconfig.py.)


## Dependencies
* Python 3. (test on 3.7.6)
* torch 1.5.1
* tqdm 4.42.1
* pytorch-transformers 1.2.0
* spacy 2.3.2


## Usage


**Pre-steps, run below code to get the data for auxiliary tasks:**

```
bash ./scripts/create_aux_data.sh
```


**To get the BERT-B-UDA results (based on bert-base-uncased) for all domain transfer pairs:**

* Step 1, run:
```
bash ./scripts/run_base_feature_learning.sh
```

* Step 2, run:
```
bash ./scripts/run_base_uda.sh
```


**To get the BERT-E-UDA results(based on bert-extented) for all domain transfer pairs:**

* Step 1, run:
```
bash ./scripts/run_extented_feature_learning.sh
```

* Step 2, run:
```
bash ./scripts/run_extented_uda.sh
```


**To get BERT-Base results, run:**

```
bash ./scripts/run_base.sh
```


**To get BERT-Extented results, run:**

```
bash ./scripts/run_extented.sh
```
