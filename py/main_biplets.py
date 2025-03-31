import os
import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import torch
from gensim.models import  KeyedVectors
import transformers

# -----------------------------------------------------

# EMBEDDINGS/MODELS
ft_spa = "py/embeddings/cc.es.300.bin" # spanish fasttext embeddings
w2v_spa = "py/embeddings/sbw_vectors.bin" # spanish word2vec (SBW) embeddings
mult_bert = "bert-base-multilingual-cased" # multilingual BERT language model

# DATASETS
um_filtered_spa = pd.read_csv("py/datasets/filtered_spa.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])

# -----------------------------------------------------

def choose_embeddings(model_name):

    # WORD2VEC SPANISH EMBEDDINGS
    if model_name.lower() in ["word2vec", "w"]:
        print('\nLoading Spanish Billion Words (Word2Vec) embeddings...')
        # load the SBW embeddings
        model = KeyedVectors.load_word2vec_format(w2v_spa, binary=True, limit=50000) # limit is for testing purposes
        return model, "Word2Vec", None

    # FASTTEXT SPANISH EMBEDDINGS
    if model_name.lower() in ["fasttext", "f"]:
        print('\nLoading FastText embeddings...')
        # check if they exist, if they don't, download them
        if not os.path.exists(ft_spa):
            fasttext.util.download_model('es', if_exists='ignore')
        model = fasttext.load_model(ft_spa)
        return model, "FastText", None

    # MULTILINGUAL BERT MODEL
    if model_name.lower() in ["bert", "b"]:
        print('\nLoading Multilingual BERT embeddings...')
        # load BERT embeddings
        tokenizer = transformers.AutoTokenizer.from_pretrained(mult_bert)
        model = transformers.AutoModel.from_pretrained(mult_bert)
        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        model.to(device)
        return model, "BERT", tokenizer

    elif model_name.lower() == "exit":
        exit()

    else:
        raise ValueError("Invalid model name.")

def calculate_sims(model, model_name, tokenizer,  data):
    # calculate cosine similarities between pivot and inflection/derivation
    print("Calculating similarities...")
    results = []
    not_found = 0 # for SBW embeddings
    for _, row in data.iterrows():
        pivot, inflection = row["pivot"], row["inflection"]

        if model_name == "FastText":
            vec_pivot = model.get_word_vector(pivot) # vector of the pivot
            vec_inflection = model.get_word_vector(inflection) # vector of the inflection

        elif model_name == "SBW":
            
            if pivot in model and inflection in model: # need to check since w2v does not use subwords
                vec_pivot = model[pivot]
                vec_inflection = model[inflection]  # Vector of the inflection

            else:
                not_found += 1

        elif model_name == "BERT" and tokenizer is not None:

            inputs = tokenizer([pivot, inflection], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # use the [CLS] token embedding as the sentence/word representation 
            vec_pivot = outputs.last_hidden_state[0][0].numpy()
            vec_inflection = outputs.last_hidden_state[1][0].numpy()
            
        # calculate similarity between pivot and inflection
        sim_inflection = 1 - cosine(vec_pivot, vec_inflection) # need 1 - cosine because cosine alone just measures distance, not similarity

        # append everything to the list
        results.append((pivot, inflection, sim_inflection))

    # can be deleted at some point, in the end the intention is to use the whole sbw model (now it's limited to 50k words)
    if model_name == "SBW":
        print(f"Number of words not found: {not_found}")

    # create a df with the results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"py/results/{model_name}_results.csv", index=False, header=["pivot", "inflection", "P-I similarity"])

    print(f"\n------ RESULTS OF {model_name.upper()} EMBEDDINGS ------\n")

    # similarities for mean
    sim_inflection_values = [r[2] for r in results]
    print(f"    Inflection mean similarity to pivot: {np.mean(sim_inflection_values):2f}")

# -------------------------------------------------------
# where functions are called

model, model_name, tokenizer = choose_embeddings()  #  change the model here: fasttext (f), word2vec (w) or bert (b)

calculate_sims(model, model_name, tokenizer, data=um_filtered_spa) # change the data here