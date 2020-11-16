# Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis

This repository contains code for our EMNLP 2020 paper: 

Unified Feature and Instance Based Domain Adaptation for Aspect-Based Sentiment Analysis



## Dependencies
-**Python 3**
-**torch 1.5.1**
-**tqdm 4.42.1**
-**pytorch-transformers 1.2.0**



## Usage

1. First step, run:
	bash ./scripts/run_feature_learning.sh
	to prepare the unlabel dataset for auxiliary tasks and do feature-learning for 10 domain transfer pairs.
	
2. Second, run:
	bash ./scripts/run_uda.sh
	to get the BERT-uda results for 10 domain transfer pairs.

3. For BERT-based baseline, run:
	bash ./scripts/run_base.sh
	to get the BERT-base results for 10 domian transfer pairs.

