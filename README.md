# Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis

This repository contains code for our EMNLP 2020 paper: 

Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis


## Datasets

The training data comes from four domains: 
Restaurant(R) 、 Laptop(L) 、 Service(S) 、 Devices(D) 

The unlabel data from the merge of training data of two domains(with 1:1 ration).

The in-domain corpus for train BERT-E is from [yelp](https://www.yelp.com/dataset/challenge) and [amazon reviews](http://jmcauley.ucsd.edu/data/amazon/links.html). 


## Dependencies
* Python 3
* torch 1.5.1
* tqdm 4.42.1
* pytorch-transformers 1.2.0
* spacy 2.3.2


## Usage

* Step 1, run:
		bash ./scripts/run_feature_learning.sh
	to prepare the unlabel dataset for auxiliary tasks and do feature-learning for 10 domain transfer pairs.
	
* Step 2, run:
		bash ./scripts/run_uda.sh
	to get the BERT-uda results for 10 domain transfer pairs.

* Step 3, For BERT-based baseline, run:
		bash ./scripts/run_base.sh
	to get the BERT-base results for 10 domian transfer pairs.

