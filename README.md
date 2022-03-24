# Discovering and measuring biases in movie descriptions
An implementation of bias detection and categorization for movie plot descriptions as
demonstrated by 
Xavier Ferrer, Tom van Nuenen, Jose M. Such and Natalia Criado
(https://arxiv.org/pdf/2008.02754.pdf)

Code is partly based on https://github.com/xfold/LanguageBiasesInReddit.

Semantically tagging biased words is done separately through http://ucrel-api.lancaster.ac.uk/usas/tagger.html
using vertical layout.

## Contents
- Bias_model: trains a word2vec or
fasttext model on the given csv-dataset

- Run_model: show results of analysis

- semantic_clusterings: gives statistics on the tagged word clusters

- clusters: files with (labeled) biased adjectives and nouns for female and male
    - gender_clusters: includes the clustered results of words biased towards or against the gender vectors
    - semantically tagged male and female words 
    - cluster results, with most common (two) labels in the cluster
