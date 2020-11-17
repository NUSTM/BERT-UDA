# Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis

This repository contains code for our EMNLP 2020 paper: 

Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis


## Datasets

<<<<<<< HEAD
The training data comes from four domains: 
Restaurant(R) 、 Laptop(L) 、 Service(S) 、 Devices(D) 

The unlabel data from the merge of training data of two domains(with 1:1 ration).

The in-domain corpus for train BERT-E is from [yelp](https://www.yelp.com/dataset/challenge) and [amazon reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). 


## Dependencies
* Python 3
=======
The training data comes from four domains: Restaurant(R) 、 Laptop(L) 、 Service(S) 、 Devices(D).  
For each domain transfer pairs, the unlabeled data come from a combination of training data from the two domains(ratio: 1:1).

The in-domain corpus(used for training BERT-E) come from [yelp](https://www.yelp.com/dataset/challenge) and [amazon reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). 

The link of BERT-E(BERT-Extented) will be uploaded soon.


## Dependencies
* Python 3. (test on 3.7.6)
>>>>>>> update
* torch 1.5.1
* tqdm 4.42.1
* pytorch-transformers 1.2.0
* spacy 2.3.2


## Usage

<<<<<<< HEAD
* Step 1, run:
		bash ./scripts/run_feature_learning.sh
	to prepare the unlabel dataset for auxiliary tasks and do feature-learning for 10 domain transfer pairs.
	
* Step 2, run:
		bash ./scripts/run_uda.sh
	to get the BERT-uda results for 10 domain transfer pairs.

* Step 3, For BERT-based baseline, run:
		bash ./scripts/run_base.sh
	to get the BERT-base results for 10 domian transfer pairs.

=======
**Pre-steps, run below code to get the data for auxiliary tasks:**

```
bash ./scripts/run_feature_learning.sh
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

**To get BERT-Base results run:**

```
bash ./scripts/run_base.sh
```

**To get BERT-Extented results run:**

```
bash ./scripts/run_extented.sh
```
>>>>>>> update
