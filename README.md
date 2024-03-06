# Transition based Dependency Parser with PyTorch
Transition-based dependency parsing is a popular technique used in natural language processing (NLP) to analyze the syntactic structure of sentences by predicting the dependency relations between words. It operates by performing a sequence of transition actions on a partially constructed dependency parse tree until a complete parse is achieved.

This repository contains the code for a Transition-based Dependency Parser implemented using PyTorch. The project focuses on transforming input datasets into tokens, converting them into embeddings using GloVe embeddings, building a neural network with torch.nn library, and evaluating the model's performance using Unlabelled Attachment Score (UAS) and Labelled Attachment Score (LAS).

## Methodology
#### Dataset Transformation: 
The raw dataset is padded with [PAD] tokens to get equal sized senteces and then transformed into tokens.
#### Embedding Generation: 
The tokens are converted into embeddings using pre-trained GloVe embeddings.
#### Neural Network Construction: 
A neural network is built using torch.nn library, consisting of three linear layers. The GloVe embeddings are fed into this network.
#### Model Evaluation: 
The model's performance is evaluated using Unlabelled Attachment Score (UAS) and Labelled Attachment Score (LAS).

## Experimentation
1) The model is trained and evaluated on four different sets of GloVe embeddings:
   
   i) GloVe 6B 50d

   ii) GloVe 6B 300d
   
   iii) GloVe 42B 300d

   iv) GloVe 840B 300d
   
2) Two approaches are employed for using the embeddings: taking the mean of all word embeddings sentence-wise and concatenating them together.

## Evaluation
### Performance Metrics:
We evaluate the performance of our transition-based dependency parser using two standard metrics:

#### Unlabelled Attachment Score (UAS): This metric measures the proportion of correctly predicted parent-child relationships in the dependency parse tree, regardless of the specific dependency label.

#### Labelled Attachment Score (LAS): This metric extends UAS by considering both the correctness of parent-child relationships and the accuracy of the dependency labels.

### Results:
After training and testing our model, we obtained the following evaluation results:

UAS: 0.76

LAS: 0.705

The above results are on the best GloVe embeddings, learning rates, and concatenated embeddings.

Check out our implementation of transition-based dependency parsing in the codebase
