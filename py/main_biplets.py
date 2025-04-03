import os
import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import torch
from gensim.models import  KeyedVectors
import transformers

from bert_embeddings import bert_text_preparation, get_bert_embeddings
from collections import OrderedDict

# -----------------------------------------------------

# EMBEDDINGS/MODELS
# FASTTEXT
spa_ft = "py/embeddings/spa/cc.es.300.bin" # spanish
pol_ft = "py/embeddings/pol/cc.pl.300.bin" # polish

# WORD2VEC
spa_w2v = "py/embeddings/spa/sbw_vectors.bin" # spanish (SBW)
pol_w2v = "py/embeddings/pol/nkjp+wiki-forms-all-300-skipg-ns.bin" # polish

# Multilingual BERT model
mult_bert = "bert-base-multilingual-cased"

# DATASETS
um_spa = pd.read_csv("py/datasets/spa/spa_filtered.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])
um_spa_small = pd.read_csv("py/datasets/spa/spa_filtered_small.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])
um_pol = pd.read_csv("py/datasets/pol/pol_filtered.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])

# -----------------------------------------------------

def choose_embeddings(model_name, language):
   
    embeddings_dict = {
        "spa": {
            "word2vec": spa_w2v,
            "fasttext": spa_ft,
            "bert": mult_bert,
        },
        "pol": {
            "word2vec": pol_w2v,
            "fasttext": pol_ft,
            "bert": mult_bert,
        }
    }

    embeddings = embeddings_dict[language.lower()][model_name.lower()]

    if language == "spa":
        language = "Spanish"
    elif language == "pol":
        language = "Polish"

    if model_name.lower() == "word2vec":
        print(f'\nLoading {language} Word2Vec embeddings...')
        # load the SBW embeddings
        model = KeyedVectors.load_word2vec_format(embeddings, binary=True)
        return model, "Word2Vec", None, language

    if model_name.lower() == "fasttext":
        print(f'\nLoading {language} FastText embeddings...')
        # check if they exist, if they don't, download them
        if not os.path.exists(embeddings):
            print(f"Downloading {language} FastText embeddings to {embeddings}...")
            fasttext.util.download_model('es')
            os.rename("cc.es.300.bin", embeddings)  # Move the downloaded file to the desired location
        model = fasttext.load_model(embeddings)
        return model, "FastText", None, language

    # MULTILINGUAL BERT MODEL. NEEDS CHANGES!
    if model_name.lower() in ["bert", "b"]:
        print('\nLoading Multilingual BERT model...')
        # load BERT embeddings
        tokenizer = transformers.AutoTokenizer.from_pretrained(mult_bert)
        model = transformers.AutoModel.from_pretrained(mult_bert, output_hidden_states=True)

        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        model.to(device)
        return model, "BERT", tokenizer, language

    elif model_name.lower() == "exit":
        exit()

    else:
        raise ValueError("Invalid model name.")

def calculate_sims(model, model_name, tokenizer, language, data):
    if language == "Spanish":
        language = "spa"
    elif language == "Polish":
        language = "pol"

    # calculate cosine similarities between pivot and inflection/derivation
    print("Calculating similarities...")
    results = []
    
    if model_name == "BERT" and tokenizer is not None:
        batch_size = 16
        for i in range(0, len(data), batch_size): # iterate over the data in batches (more efficient for BERT)
            batch = data.iloc[i:i+batch_size]
            pivots = batch["pivot"].tolist()
            inflections = batch["inflection"].tolist()

            # tokenize and encode inputs (pivots and inflections)
            inputs = tokenizer(pivots + inflections, return_tensors="pt", padding=True, truncation=True) # tokenizer returns a dictionary with input_ids, attention_mask...
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            # pass the inputs through the model
            with torch.no_grad(): # no grad because we are not training the model
                outputs = model(**inputs)
            
            # extract all hidden states
            hidden_states = outputs.hidden_states  # a tuple containing the hidden states for all the layers, each hidden state is a tensor of shape (batch_size, sequence_length, hidden_size)

            token_embeddings = torch.stack(hidden_states[-4:])  # take the last 4 layers and stack them into a single tensor of shape (4, batch_size, sequence_length, hidden_size)
            token_embeddings = torch.sum(token_embeddings, dim=0)  # sum the embeddings across the 4 layers resulting in 1 tensor of shape (batch_size, sequence_length, hidden_size)

            batch_size = len(pivots)
            # extract embeddings and convert them to np arrays
            pivot_embeddings = token_embeddings[:batch_size, 0].cpu().numpy() # get the embedding of the first token (CLS) of the first element (pivot)
            inflection_embeddings = token_embeddings[:batch_size, 0].cpu().numpy()

            # calculate similarities
            for pivot, inflection, pivot_embedding, inflection_embedding in zip(pivot, inflections, pivot_embeddings, inflection_embeddings):
                sim_inflection = 1 - cosine(pivot_embedding, inflection_embedding)
                results.append(pivot, inflection, sim_inflection)
    else:
        not_found = 0 # for Word2Vec embeddings
        for _, row in data.iterrows():
            pivot, inflection = row["pivot"], row["inflection"]

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
            sim_inflection = 1 - cosine(pivot_embedding, inflection_embedding) # need 1 - cosine because cosine alone just measures distance, not similarity

            # append everything to the results list
            results.append((pivot, inflection, sim_inflection))

        if model_name == "Word2Vec":
            print(f"Number of words not found in Word2Vec model: {not_found}")

    # create a df with the results list
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"py/results/{language}/{language}_{model_name.lower()}_results.csv", index=False, header=["pivot", "inflection", "P-I similarity"])
    print("Results by row saved!")

    # similarities for mean
    sim_inflection_values = [r[2] for r in results]
    print(f"\n{model_name.upper()} EMBEDDINGS IN {language.upper()}")
    print(f"    MEAN SIMILARITY (VERB-INFLECTION): {np.mean(sim_inflection_values):2f}")

# -------------------------------------------------------
# where functions are called

'''
choose_embeddings takes an argument of the model name (fasttext, word2vec or bert) and a language argument (spa or pol), it outputs the model, the model name, the tokenizer (if applicable) and the language (to be used in the results file name and file path).

calculate_sims takes what choose_embeddings outputs and an argument of the file to be used.
'''

# # FASTTEXT
# model, model_name, tokenizer, language = choose_embeddings("fasttext", language="spa") # SPANISH
# calculate_sims(model, model_name, tokenizer, language, data=um_spa)

# model, model_name, tokenizer, language = choose_embeddings("fasttext", "pol") # POLISH
# calculate_sims(model, model_name, tokenizer, language, um_pol)

# # WORD2VEC
# model, model_name, tokenizer, language = choose_embeddings("word2vec", "spa") # SPANISH
# calculate_sims(model, model_name, tokenizer, language, um_spa)

# model, model_name, tokenizer, language = choose_embeddings("word2vec", "pol") # POLISH
# calculate_sims(model, model_name, tokenizer, language, um_pol)

# MULTILINGUAL BERT
# bert takes a really long time
model, model_name, tokenizer, language = choose_embeddings("bert", "spa") # SPANISH
calculate_sims(model, model_name, tokenizer, language, um_spa_small)