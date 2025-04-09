library(ggplot2)
library(dplyr)

# group_by(spa_ft_inf, pivot)
# filter(spa_ft_der, affix == "-ito") # filter by affix
# filter(spa_ft_der, affix %in% "-ito")
# arrange(spa_ft_der, similarity) / desc(similarity)
# select() # to select columns/column names
# new_df = select(spa_ft_der, pivot, derivation, affix)
# select(spa_ft_der, pivot:similarity)
# %>% "and then"


# FASTTEXT
# INFLECTION
spa_ft_inf <- read.csv("results/spa/spa_fasttext_inflection_results.csv")
pol_ft_inf <- read.csv("results/pol/pol_fasttext_inflection_results.csv")
# DERIVATION
spa_ft_der <- read.csv("results/spa/spa_fasttext_derivation_results.csv")
pol_ft_der <- read.csv("results/pol/pol_fasttext_derivation_results.csv")

# WORD2VEC
# INFLECTION
spa_w2v_inf <- read.csv("results/spa/spa_word2vec_inflection_results.csv")
pol_w2v_inf <- read.csv("results/pol/pol_word2vec_inflection_results.csv")
# DERIVATION
spa_w2v_der <- read.csv("results/spa/spa_word2vec_derivation_results.csv")
pol_w2v_der <- read.csv("results/pol/pol_word2vec_derivation_results.csv")

# BERT
# INFLECTION
spa_bert_inf <- read.csv("results/spa/spa_bert_inflection_results.csv")
pol_bert_inf <- read.csv("results/pol/pol_bert_inflection_results.csv")
# DERIVATION
spa_bert_der <- read.csv("results/spa/spa_bert_derivation_results.csv")
pol_bert_der <- read.csv("results/pol/pol_bert_derivation_results.csv")

# adding model, language and type (inf or der) to the dataset in order to plot it
spa_ft_inf <- spa_ft_inf %>%
  mutate(model = "FastText", language = "Spanish", process = "Inflection", category = "V:V")

pol_ft_inf <- pol_ft_inf %>%
  mutate(model = "FastText", language = "Polish", process = "Inflection", category = "V:V")

spa_ft_der <- spa_ft_der %>%
  mutate(model = "FastText", language = "Spanish", process = "Derivation")

pol_ft_der <- pol_ft_der %>%
  mutate(model = "FastText", language = "Polish", process = "Derivation")

spa_w2v_inf <- spa_w2v_inf %>%
  mutate(model = "Word2Vec", language = "Spanish", process = "Inflection", category = "V:V")

pol_w2v_inf <- pol_w2v_inf %>%
  mutate(model = "Word2Vec", language = "Polish", process = "Inflection", category = "V:V")

spa_w2v_der <- spa_w2v_der %>%
  mutate(model = "Word2Vec", language = "Spanish", process = "Derivation")

pol_w2v_der <- pol_w2v_der %>%
  mutate(model = "Word2Vec", language = "Polish", process = "Derivation")

spa_bert_inf <- spa_bert_inf %>%
  mutate(model = "Multilingual BERT", language = "Spanish", process = "Inflection", category = "V:V")

pol_bert_inf <- pol_bert_inf %>%
  mutate(model = "Multilingual BERT", language = "Polish", process = "Inflection", category = "V:V")

spa_bert_der <- spa_bert_der %>%
  mutate(model = "Multilingual BERT", language = "Spanish", process = "Derivation")

pol_bert_der <- pol_bert_der %>%
  mutate(model = "Multilingual BERT", language = "Polish", process = "Derivation")


# first plot
# group all the dataframes into one
all_derivation <- bind_rows(spa_ft_der, pol_ft_der, spa_w2v_der, 
pol_w2v_der, spa_bert_der, pol_bert_der)

# group by category, model and language and get the mean similarity of each 
# i.e.:       category   model   language      mean_similarity
#              ADJ:ADJ    FasText   Polish        0.6197
mean_by_category <- all_derivation %>%
  group_by(category, model, language) %>%
  summarise(mean_similarity = mean(similarity, na.rm = TRUE), .groups = 'drop')

# add a new column with "same category" or "different cattegory" 
mean_by_category <- mean_by_category %>%
  mutate(category_group = ifelse(gsub(":.*", "", category) == gsub(".*:", "", category), "Same category", "Different category"))

# plot the df
plot1 <- ggplot(mean_by_category, aes(x = category, y = mean_similarity, 
color = model)) +
  geom_point(size = 4, position = position_dodge(width = 0.5), na.rm = TRUE) +
  labs(x = "Category", y = "Mean similarity", 
  color = "Embeddings model", shape = "Language", title = "Same category vs. different category in derivation") +
  facet_wrap(language ~ category_group, scales = "free_x") +
  scale_x_discrete(drop = TRUE) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ggsave("C:/PythonCode/TFM_GLS/plot1.pdf",plot= plot1)

all_inflection <- bind_rows(spa_ft_inf, pol_ft_inf, spa_w2v_inf, 
pol_w2v_inf, spa_bert_inf, pol_bert_inf)



# # second plot
# all_data <- bind_rows(spa_ft_inf, pol_ft_inf, spa_ft_der, pol_ft_der, spa_w2v_inf, pol_w2v_inf, spa_w2v_der, pol_w2v_der, spa_bert_inf, pol_bert_inf, spa_bert_der, pol_bert_der)

# mean_by_category <- all_data %>%
#   group_by(category, model, language, process) %>%
#   summarise(mean_similarity = mean(similarity, na.rm = TRUE), .groups = 'drop')

# mean_by_category <- mean_by_category %>%
#   mutate(category_group = ifelse(gsub(":.*", "", category) == gsub(".*:", "", category), "Same category", "Different category"))

# plot2 <- ggplot(mean_by_category, aes(x = category, y = mean_similarity, color = model, shape = language)) +
#   geom_point(size = 2, position = position_dodge(width = 0.5), na.rm = TRUE) +
#   labs(x = "Category", y = "Mean Similarity", color = "Embeddings model", shape = "Language") +
#   facet_wrap(~category_group, scales="free_x") +
#   scale_x_discrete(drop = TRUE) +
#   theme_light() +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1))
