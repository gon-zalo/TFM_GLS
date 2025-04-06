library(dplyr)
library(ggplot2)

# FASTTEXT
spa_ft_results <- read.csv("py/results/spa/spa_fasttext_results.csv")
pol_ft_results <- read.csv("py/results/pol/pol_fasttext_results.csv")

# WORD2VEC
spa_w2v_results <- read.csv("py/results/spa/spa_word2vec_results.csv")
pol_w2v_results <- read.csv("py/results/pol/pol_word2vec_results.csv")

# BERT
spa_bert_results <- read.csv("py/results/spa/spa_bert_results.csv")
pol_bert_results <- read.csv("py/results/pol/pol_bert_results.csv")