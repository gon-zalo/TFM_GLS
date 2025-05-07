# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import torch
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModelWithLMHead
from gensim.models import  KeyedVectors
import transformers
from filter_unimorph import filter_inflection, filter_derivation

# -----------------------------------------------------

# EMBEDDINGS/MODELS
# FASTTEXT
spa_ft = "embeddings/spa/cc.es.300.bin" # spanish
pol_ft = "embeddings/pol/cc.pl.300.bin" # polish

# WORD2VEC
spa_w2v = "embeddings/spa/sbw_vectors.bin" # spanish (SBW)
pol_w2v = "embeddings/pol/nkjp+wiki-forms-all-300-skipg-ns.bin" # polish

# BERT models
mult_bert = "bert-base-multilingual-cased" # multilingual bert
polbert = "dkleczek/bert-base-polish-cased-v1"
roberta = "skimai/spanberta-base-cased"

# DATASETS
    # INFLECTION
spa_inf = pd.read_csv("datasets/spa/spa_inflections.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])
pol_inf = pd.read_csv("datasets/pol/pol_inflections.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])

    # SHUFFLED INFLECTION
spa_inf_shuf = pd.read_csv("datasets/spa/spa_inflections_shuffled.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])
pol_inf_shuf = pd.read_csv("datasets/pol/pol_inflections_shuffled.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])

        # SUBSET INFLECTION
spa_inf_subs = pd.read_csv("datasets/spa/spa_inflections_subset.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])
pol_inf_subs = pd.read_csv("datasets/pol/pol_inflections_subset.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])


    # DERIVATION
spa_der = pd.read_csv("datasets/spa/spa_derivations.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])
pol_der = pd.read_csv("datasets/pol/pol_derivations.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])

    # SHUFFLED DERIVATION
spa_der_shuf = pd.read_csv("datasets/spa/spa_derivations_shuffled.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])
pol_der_shuf = pd.read_csv("datasets/pol/pol_derivations_shuffled.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])

        # SUBSET DERIVATION
spa_der_subs = pd.read_csv("datasets/spa/spa_derivations_subset.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])
pol_der_subs = pd.read_csv("datasets/pol/pol_derivations_subset.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])


# -----------------------------------------------------

def choose_embeddings(model_name, language):
   
    embeddings_dict = {
        "spa": {
            "word2vec": spa_w2v,
            "fasttext": spa_ft,
            "bert": mult_bert,
            "roberta":  roberta,
        },
        "pol": {
            "word2vec": pol_w2v,
            "fasttext": pol_ft,
            "bert": mult_bert,
            "polbert": polbert,
        }
    }

    embeddings = embeddings_dict[language.lower()][model_name.lower()]

    if language == "spa":
        language = "Spanish"
    elif language == "pol":
        language = "Polish"

    #  WORD2VEC
    if model_name.lower() == "word2vec":
        print(f'\nLoading {language} Word2Vec embeddings...')
        # load the SBW embeddings
        model = KeyedVectors.load_word2vec_format(embeddings, binary=True)
        return model, "Word2Vec", None, language

    # FASTTEXT
    if model_name.lower() == "fasttext":
        print(f'\nLoading {language} FastText embeddings...')
        # check if they exist, if they don't, download them
        if not os.path.exists(embeddings):
            print(f"Downloading {language} FastText embeddings to {embeddings}...")
            fasttext.util.download_model('es')
            os.rename("cc.es.300.bin", embeddings)  # Move the downloaded file to the desired location
        model = fasttext.load_model(embeddings)
        return model, "FastText", None, language

    # BERT models
    if model_name.lower() == "polbert":
        print('\nLoading PolBERT model...')
        # load BERT embeddings
        tokenizer = transformers.AutoTokenizer.from_pretrained(embeddings) # and
        model = transformers.AutoModel.from_pretrained(embeddings, output_hidden_states=True)

        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        model.to(device)

        return model, "PolBERT", tokenizer, language

    if model_name.lower() == "roberta":
        print('\nLoading RoBERTa model...')
        tokenizer = transformers.AutoTokenizer.from_pretrained(embeddings)
        model = transformers.AutoModelWithLMHead.from_pretrained(embeddings, output_hidden_states = True)

        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        model.to(device)

        return model, "RoBERTa", tokenizer, language

    else:
        raise ValueError("Invalid model name.")

def sim_inflection(model, model_name, tokenizer, language, data):
    if language == "Spanish":
        language = "spa"
    elif language == "Polish":
        language = "pol"

    # Calculate cosine similarities between pivot and inflection
    print("Calculating similarities...")
    results = []

    if model_name == "RoBERTa" or "PolBERT" and tokenizer is not None:
        batch_size = 16
        for i in range(0, len(data), batch_size):  # iterate over the data in batches (more efficient for BERT)
            batch = data.iloc[i:i+batch_size]
            pivots = batch["pivot"].tolist()
            inflections = batch["inflection"].tolist()
            categories = batch["category"].tolist()

            # tokenize and encode inputs (pivots and inflections)
            inputs = tokenizer(pivots + inflections, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            # pass the inputs through the model
            with torch.no_grad(): # no grad because we are not training the model
                outputs = model(**inputs)

            # extract all hidden states
            hidden_states = outputs.hidden_states  # Tuple of hidden states for all layers

            # stack and sum the last 4 layers
            token_embeddings = torch.stack(hidden_states[-4:], dim=0)  # take the last 4 layers and stack them into a single tensor of shape (4, batch_size, sequence_length, hidden_size
            token_embeddings = torch.sum(token_embeddings, dim=0)  # sum the embeddings across the 4 layers resulting in 1 tensor of shape (batch_size, sequence_length, hidden_size)

            # split token_embeddings into pivot and inflection batches
            batch_size = len(pivots)
            pivot_embeddings = token_embeddings[:batch_size]  # embeddings for pivots
            inflection_embeddings = token_embeddings[batch_size:]  # embeddings for inflections

            # process each pivot-inflection pair
            for pivot, inflection, pivot_embedding, inflection_embedding, category in zip(
                pivots, inflections, pivot_embeddings, inflection_embeddings, categories
            ):
                # tokenize pivot and inflection
                pivot_tokens = tokenizer.tokenize(pivot)
                inflection_tokens = tokenizer.tokenize(inflection)

                # extract subword embeddings and average them
                pivot_subword_embeddings = pivot_embedding[1:len(pivot_tokens)+1, :]  # exclude [CLS] and [SEP]
                inflection_subword_embeddings = inflection_embedding[1:len(inflection_tokens)+1, :]  # exclude [CLS] and [SEP]

                pivot_word_embedding = torch.mean(pivot_subword_embeddings, dim=0)  # average subword embeddings
                inflection_word_embedding = torch.mean(inflection_subword_embeddings, dim=0)  # average subword embeddings

                # normalize embeddings
                pivot_word_embedding = normalize(pivot_word_embedding, dim=0)
                inflection_word_embedding = normalize(inflection_word_embedding, dim=0)

                # calculate cosine similarity
                similarity = 1 - cosine(pivot_word_embedding.cpu().numpy(), inflection_word_embedding.cpu().numpy())
                results.append((pivot, inflection, similarity, category))

    else:
        not_found = 0 # for Word2Vec embeddings
        for _, row in data.iterrows():
            pivot, inflection, category = row["pivot"], row["inflection"], row["category"]

            if model_name == "FastText":
                pivot_embedding = model.get_word_vector(pivot) # vector of the pivot
                inflection_embedding = model.get_word_vector(inflection) # vector of the inflection

            elif model_name == "Word2Vec":
                if pivot in model and inflection in model: # need to check since w2v does not use subwords
                    pivot_embedding = model[pivot]
                    inflection_embedding = model[inflection]  # Vector of the inflection
                else:
                    not_found += 1
                    continue

            # calculate similarity between pivot and inflection
            similarity = 1 - cosine(pivot_embedding, inflection_embedding) # need 1 - cosine because cosine alone just measures distance, not similarity

            # append everything to the results list
            results.append((pivot, inflection, similarity, category))

        if model_name == "Word2Vec":
            print(f"Number of words not found in Word2Vec model: {not_found}")

    # create a dataFrame with the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/{language}/{language}_{model_name.lower()}_inflection_shuffled_results.csv", index=False, header=["pivot", "inflection", "similarity", "category"])
    print("Results by row saved!")

    # calculate and print the mean similarity
    similarity_values = [r[2] for r in results]
    print(f"\n{model_name.upper()} EMBEDDINGS IN {language.upper()}")
    print(f"    MEAN SIMILARITY (PIVOT-INFLECTION): {np.mean(similarity_values):.2f}")

# function to shuffle inflections and extract the mean similarity of each loop
def shuffle_inflection(model, model_name, tokenizer, language, data, tense):
    if language == "Spanish":
        language = "Spanish"
        lang = "spa"
    elif language == "Polish":
        language == "Polish"
        lang = "pol"

    # Calculate cosine similarities between pivot and inflection
    print("Calculating similarities...")

    mean_similarity_values = [] # empty list to store the mean of each loop

    for number in range(1,101): # shuffle data
        print(f"Shuffle #{number}")
    
        data[["inflection", "category"]] = data[["inflection", "category"]].sample(frac=1, random_state=123).reset_index(drop=True)

        # this empties both lists at the start of each loop
        results = [] # empty list to store all the columns of the df
        similarity_values = [] # empty list to store the similarity values of the whole dataframe

        # same code as sim_inflection and sim_derivation
        if model_name == "RoBERTa" or "PolBERT" and tokenizer is not None:
            batch_size = 16
            for i in range(0, len(data), batch_size):  # iterate over the data in batches (more efficient for BERT)
                batch = data.iloc[i:i+batch_size]
                pivots = batch["pivot"].tolist()
                inflections = batch["inflection"].tolist()
                categories = batch["category"].tolist()

                # tokenize and encode inputs (pivots and inflections)
                inputs = tokenizer(pivots + inflections, return_tensors="pt", padding=True, truncation=True)
                inputs = {key: value.to(model.device) for key, value in inputs.items()}

                # pass the inputs through the model
                with torch.no_grad(): # no grad because we are not training the model
                    outputs = model(**inputs)

                # extract all hidden states
                hidden_states = outputs.hidden_states  # Tuple of hidden states for all layers

                # stack and sum the last 4 layers
                token_embeddings = torch.stack(hidden_states[-4:], dim=0)  # take the last 4 layers and stack them into a single tensor of shape (4, batch_size, sequence_length, hidden_size
                token_embeddings = torch.sum(token_embeddings, dim=0)  # sum the embeddings across the 4 layers resulting in 1 tensor of shape (batch_size, sequence_length, hidden_size)

                # split token_embeddings into pivot and inflection batches
                batch_size = len(pivots)
                pivot_embeddings = token_embeddings[:batch_size]  # embeddings for pivots
                inflection_embeddings = token_embeddings[batch_size:]  # embeddings for inflections

                # process each pivot-inflection pair
                for pivot, inflection, pivot_embedding, inflection_embedding, category in zip(
                    pivots, inflections, pivot_embeddings, inflection_embeddings, categories
                ):
                    # tokenize pivot and inflection
                    pivot_tokens = tokenizer.tokenize(pivot)
                    inflection_tokens = tokenizer.tokenize(inflection)

                    # extract subword embeddings and average them
                    pivot_subword_embeddings = pivot_embedding[1:len(pivot_tokens)+1, :]  # exclude [CLS] and [SEP]
                    inflection_subword_embeddings = inflection_embedding[1:len(inflection_tokens)+1, :]  # exclude [CLS] and [SEP]

                    pivot_word_embedding = torch.mean(pivot_subword_embeddings, dim=0)  # average subword embeddings
                    inflection_word_embedding = torch.mean(inflection_subword_embeddings, dim=0)  # average subword embeddings

                    # normalize embeddings
                    pivot_word_embedding = normalize(pivot_word_embedding, dim=0)
                    inflection_word_embedding = normalize(inflection_word_embedding, dim=0)

                    # calculate cosine similarity
                    similarity = 1 - cosine(pivot_word_embedding.cpu().numpy(), inflection_word_embedding.cpu().numpy())
                    results.append((pivot, inflection, similarity, category))

        else:
            for _, row in data.iterrows():
                pivot, inflection, category = row["pivot"], row["inflection"], row["category"]

                if model_name == "FastText":
                    pivot_embedding = model.get_word_vector(pivot) # vector of the pivot
                    inflection_embedding = model.get_word_vector(inflection) # vector of the inflection

                elif model_name == "Word2Vec":
                    if pivot in model and inflection in model: # need to check since w2v does not use subwords
                        pivot_embedding = model[pivot]
                        inflection_embedding = model[inflection]  # Vector of the inflection
                    else:
                        continue

                # calculate similarity between pivot and inflection
                similarity = 1 - cosine(pivot_embedding, inflection_embedding) # need 1 - cosine because cosine alone just measures distance, not similarity

                # append everything to the results list
                results.append((pivot, inflection, similarity, category))
            
        similarity_values = [r[2] for r in results]
        # print(similarity_values[:15])
        # print(len(similarity_values))
        mean_similarity_values.append(np.mean(similarity_values)) # append the mean of all the similarity values of the loop

    mean_similarity_values = pd.DataFrame(mean_similarity_values, columns=['mean_similarity'])

    mean_similarity_values['model'] = f"{model_name}"
    mean_similarity_values['language'] = f"{language}"
    mean_similarity_values['type'] = "Inflection"
    mean_similarity_values['category'] = f"{tense.capitalize()}"

    mean_similarity_values.to_csv(f"results/baseline/{lang}_{model_name.lower()}_inflection_{tense}.csv", index=False)
    print("Mean similarity values saved!")

def sim_derivation(model, model_name, tokenizer, language, data):
    if language == "Spanish":
        language = "spa"
    elif language == "Polish":
        language = "pol"

    # calculate cosine similarities between pivot and derivation
    print("Calculating similarities...")
    results = []

    if model_name == "RoERTa" or "PolBERT" and tokenizer is not None:
        batch_size = 16
        for i in range(0, len(data), batch_size):  # iterate over the data in batches
            batch = data.iloc[i:i+batch_size]
            pivots = batch["pivot"].tolist()
            derivations = batch["derivation"].tolist()
            categories = batch["category"].tolist()
            affixes = batch["affix"].tolist()

            # tokenize and encode inputs (pivots and derivations)
            inputs = tokenizer(pivots + derivations, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            # pass the inputs through the model
            with torch.no_grad():
                outputs = model(**inputs)

            # extract all hidden states
            hidden_states = outputs.hidden_states  # tuple of hidden states for all layers

            # stack and sum the last 4 layers
            token_embeddings = torch.stack(hidden_states[-4:], dim=0)  # shape: (4, batch_size, seq_length, hidden_size)
            token_embeddings = torch.sum(token_embeddings, dim=0)  # shape: (batch_size, seq_length, hidden_size)

            # split token_embeddings into pivot and derivation batches
            batch_size = len(pivots)
            pivot_embeddings = token_embeddings[:batch_size]  # embeddings for pivots
            derivation_embeddings = token_embeddings[batch_size:]  # embeddings for derivations

            # process each pivot-derivation pair
            for pivot, derivation, pivot_embedding, derivation_embedding, category, affix in zip(
                pivots, derivations, pivot_embeddings, derivation_embeddings, categories, affixes
            ):
                # tokenize pivot and derivation
                pivot_tokens = tokenizer.tokenize(pivot)
                derivation_tokens = tokenizer.tokenize(derivation)

                # extract subword embeddings and average them
                pivot_subword_embeddings = pivot_embedding[1:len(pivot_tokens)+1, :]  # exclude [CLS] and [SEP]
                derivation_subword_embeddings = derivation_embedding[1:len(derivation_tokens)+1, :]  # exclude [CLS] and [SEP]

                pivot_word_embedding = torch.mean(pivot_subword_embeddings, dim=0)  # average subword embeddings
                derivation_word_embedding = torch.mean(derivation_subword_embeddings, dim=0)  # average subword embeddings

                # normalize embeddings
                pivot_word_embedding = normalize(pivot_word_embedding, dim=0)
                derivation_word_embedding = normalize(derivation_word_embedding, dim=0)

                # calculate cosine similarity
                similarity = 1 - cosine(pivot_word_embedding.cpu().numpy(), derivation_word_embedding.cpu().numpy())
                results.append((pivot, derivation, similarity, category, affix))

    else:
        not_found = 0 # for Word2Vec embeddings
        for _, row in data.iterrows():
            pivot, derivation, category, affix = row["pivot"], row["derivation"], row["category"], row["affix"]

            if model_name == "FastText":
                pivot_embedding = model.get_word_vector(pivot) # vector of the pivot
                derivation_embedding = model.get_word_vector(derivation) # vector of the inflection

            elif model_name == "Word2Vec":
                if pivot in model and derivation in model: # need to check since w2v does not use subwords
                    pivot_embedding = model[pivot]
                    derivation_embedding = model[derivation]  # Vector of the inflection
                else:
                    not_found += 1
                    continue

            # calculate similarity between pivot and inflection
            similarity = 1 - cosine(pivot_embedding, derivation_embedding) # need 1 - cosine because cosine alone just measures distance, not similarity

            # append everything to the results list
            results.append((pivot, derivation, similarity, category, affix))

        if model_name == "Word2Vec":
            print(f"Number of words not found in Word2Vec model: {not_found}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/{language}/{language}_{model_name.lower()}_derivation_results.csv", index=False, header=["pivot", "derivation", "similarity", "category", "affix"])

    # Calculate and print the mean similarity
    similarity_values = [r[2] for r in results]
    print(f"\n{model_name.upper()} EMBEDDINGS IN {language.upper()}")
    print(f"    MEAN SIMILARITY (PIVOT-DERIVATION): {np.mean(similarity_values):.2f}")

def shuffle_derivation(model, model_name, tokenizer, language, data, name):
     # same as shuffle_inflection but for derivation dataset
    if language == "Spanish":
        language = "Spanish"
        lang = "spa"
    elif language == "Polish":
        language == "Polish"
        lang = "pol"

    # Calculate cosine similarities between pivot and derivation
    print("Calculating similarities...")

    mean_similarity_values = [] # empty list to store the mean of each loop

    for number in range(1,101): # shuffle data
        print(f"Shuffle #{number}")
    
        data[["derivation", "category", "affix"]] = data[["derivation", "category", "affix"]].sample(frac=1, random_state=123).reset_index(drop=True)

        # this empties both lists at the start of each loop
        results = [] # empty list to store all the columns of the df
        similarity_values = [] # empty list to store the similarity values of the whole dataframe

        # same code as sim_inflection and sim_derivation
        if model_name == "RoERTa" or "PolBERT" and tokenizer is not None:
            batch_size = 16
            for i in range(0, len(data), batch_size):  # iterate over the data in batches
                batch = data.iloc[i:i+batch_size]
                pivots = batch["pivot"].tolist()
                derivations = batch["derivation"].tolist()
                categories = batch["category"].tolist()
                affixes = batch["affix"].tolist()

                # tokenize and encode inputs (pivots and derivations)
                inputs = tokenizer(pivots + derivations, return_tensors="pt", padding=True, truncation=True)
                inputs = {key: value.to(model.device) for key, value in inputs.items()}

                # pass the inputs through the model
                with torch.no_grad():
                    outputs = model(**inputs)

                # extract all hidden states
                hidden_states = outputs.hidden_states  # tuple of hidden states for all layers

                # stack and sum the last 4 layers
                token_embeddings = torch.stack(hidden_states[-4:], dim=0)  # shape: (4, batch_size, seq_length, hidden_size)
                token_embeddings = torch.sum(token_embeddings, dim=0)  # shape: (batch_size, seq_length, hidden_size)

                # split token_embeddings into pivot and derivation batches
                batch_size = len(pivots)
                pivot_embeddings = token_embeddings[:batch_size]  # embeddings for pivots
                derivation_embeddings = token_embeddings[batch_size:]  # embeddings for derivations

                # process each pivot-derivation pair
                for pivot, derivation, pivot_embedding, derivation_embedding, category, affix in zip(
                    pivots, derivations, pivot_embeddings, derivation_embeddings, categories, affixes
                ):
                    # tokenize pivot and derivation
                    pivot_tokens = tokenizer.tokenize(pivot)
                    derivation_tokens = tokenizer.tokenize(derivation)

                    # extract subword embeddings and average them
                    pivot_subword_embeddings = pivot_embedding[1:len(pivot_tokens)+1, :]  # exclude [CLS] and [SEP]
                    derivation_subword_embeddings = derivation_embedding[1:len(derivation_tokens)+1, :]  # exclude [CLS] and [SEP]

                    pivot_word_embedding = torch.mean(pivot_subword_embeddings, dim=0)  # average subword embeddings
                    derivation_word_embedding = torch.mean(derivation_subword_embeddings, dim=0)  # average subword embeddings

                    # normalize embeddings
                    pivot_word_embedding = normalize(pivot_word_embedding, dim=0)
                    derivation_word_embedding = normalize(derivation_word_embedding, dim=0)

                    # calculate cosine similarity
                    similarity = 1 - cosine(pivot_word_embedding.cpu().numpy(), derivation_word_embedding.cpu().numpy())
                    results.append((pivot, derivation, similarity, category, affix))

        else:
            not_found = 0 # for Word2Vec embeddings
            for _, row in data.iterrows():
                pivot, derivation, category, affix = row["pivot"], row["derivation"], row["category"], row["affix"]

                if model_name == "FastText":
                    pivot_embedding = model.get_word_vector(pivot) # vector of the pivot
                    derivation_embedding = model.get_word_vector(derivation) # vector of the inflection

                elif model_name == "Word2Vec":
                    if pivot in model and derivation in model: # need to check since w2v does not use subwords
                        pivot_embedding = model[pivot]
                        derivation_embedding = model[derivation]  # Vector of the inflection
                    else:
                        not_found += 1
                        continue

                # calculate similarity between pivot and inflection
                similarity = 1 - cosine(pivot_embedding, derivation_embedding) # need 1 - cosine because cosine alone just measures distance, not similarity

                # append everything to the results list
                results.append((pivot, derivation, similarity, category, affix))

        similarity_values = [r[2] for r in results]
        # print(similarity_values[:15])
        # print(len(similarity_values))
        mean_similarity_values.append(np.mean(similarity_values)) # append the mean of all the similarity values of the loop

    mean_similarity_values = pd.DataFrame(mean_similarity_values, columns=['mean_similarity'])

    mean_similarity_values['model'] = f"{model_name}"
    mean_similarity_values['language'] = f"{language}"
    mean_similarity_values['type'] = "Derivation"
    mean_similarity_values['category'] = f"{category}"

    mean_similarity_values.to_csv(f"results/baseline/{lang}_{model_name.lower()}_derivation_{name.replace(':', '_')}.csv", index=False)
    print("Mean similarity values saved!")

# -------------------------------------------------------
# where functions are called

'''
choose_embeddings takes an argument of the model name (fasttext, word2vec or bert) and a language argument (spa or pol), it outputs the model, the model name, the tokenizer (if applicable) and the language (to be used in the results file name and file path).

calculate_sims takes what choose_embeddings outputs and an argument of the file to be used.
'''


###### INFLECTION ######
# # FASTTEXT
# model, model_name, tokenizer, language = choose_embeddings("fasttext", language="spa") # SPANISH
# sim_inflection(model, model_name, tokenizer, language, data=spa_inf_shuf)

# model, model_name, tokenizer, language = choose_embeddings("fasttext", "pol") # POLISH
# sim_inflection(model, model_name, tokenizer, language, pol_inf_shuf)

# # WORD2VEC
# model, model_name, tokenizer, language = choose_embeddings("word2vec", "spa") # SPANISH
# sim_inflection(model, model_name, tokenizer, language, spa_inf_shuf)

# model, model_name, tokenizer, language = choose_embeddings("word2vec", "pol") # POLISH
# sim_inflection(model, model_name, tokenizer, language, pol_inf_shuf)

# ###### DERIVATION ######
# # FASTTEXT
# model, model_name, tokenizer, language = choose_embeddings("fasttext", "spa")
# sim_derivation(model, model_name, tokenizer, language, spa_der_shuf)

# model, model_name, tokenizer, language = choose_embeddings("fasttext", "pol")
# sim_derivation(model, model_name, tokenizer, language, pol_der_shuf)

# # WORD2VEC
# model, model_name, tokenizer, language = choose_embeddings("word2vec", "spa")
# sim_derivation(model, model_name, tokenizer, language, spa_der_shuf)

# model, model_name, tokenizer, language = choose_embeddings("word2vec", "pol")
# sim_derivation(model, model_name, tokenizer, language, pol_der_shuf)

# # BERT (better in google colab)
# # bert takes a really long time
# model, model_name, tokenizer, language = choose_embeddings("bert", "spa") # SPANISH
# sim_inflection(model, model_name, tokenizer, language, um_spa_small) # small dataset to test

####### BASELINE FUNCTIONS #######

# DERIVATION
# spanish
# filtered_dfs = filter_derivation(spa_der, 'spa')
# model, model_name, tokenizer, language = choose_embeddings("fasttext", "spa")
# for name, dataframe in filtered_dfs.items():
#     shuffle_derivation(model, model_name, tokenizer, language, dataframe, name)

# model, model_name, tokenizer, language = choose_embeddings("word2vec", "spa")
# for name, dataframe in filtered_dfs.items():
#     shuffle_derivation(model, model_name, tokenizer, language, dataframe, name)

# filtered_dfs = filter_derivation(pol_der, 'pol')
# model, model_name, tokenizer, language = choose_embeddings("fasttext", "pol")
# for name, dataframe in filtered_dfs.items():
#     shuffle_derivation(model, model_name, tokenizer, language, dataframe, name)

# model, model_name, tokenizer, language = choose_embeddings("word2vec", "pol")
# for name, dataframe in filtered_dfs.items():
#     shuffle_derivation(model, model_name, tokenizer, language, dataframe, name)


# INFLECTION
# present_df, past_df, future_df = filter_inflection(spa_inf, 'spa')
# model, model_name, tokenizer, language = choose_embeddings("fasttext", "spa")
# shuffle_inflection(model, model_name, tokenizer, language, present_df, "present")
# shuffle_inflection(model, model_name, tokenizer, language, past_df, 'past')
# shuffle_inflection(model, model_name, tokenizer, language, future_df, 'future')

# model, model_name, tokenizer, language = choose_embeddings("word2vec", "spa")
# shuffle_inflection(model, model_name, tokenizer, language, present_df, "present")
# shuffle_inflection(model, model_name, tokenizer, language, past_df, 'past')
# shuffle_inflection(model, model_name, tokenizer, language, future_df, 'future')

# polish
# present_df, past_df, future_df = filter_inflection(pol_inf, 'pol')
# model, model_name, tokenizer, language = choose_embeddings("fasttext", "pol")
# shuffle_inflection(model, model_name, tokenizer, language, present_df, "present")
# shuffle_inflection(model, model_name, tokenizer, language, past_df, 'past')
# shuffle_inflection(model, model_name, tokenizer, language, future_df, 'future')
# model, model_name, tokenizer, language = choose_embeddings("word2vec", "pol")
# shuffle_inflection(model, model_name, tokenizer, language, present_df, "present")
# shuffle_inflection(model, model_name, tokenizer, language, past_df, 'past')
# shuffle_inflection(model, model_name, tokenizer, language, future_df, 'future')