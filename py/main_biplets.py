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
um_pol = pd.read_csv("py/datasets/pol/pol_filtered.txt", sep="\t", header=None, names=["pivot", "inflection", "category"], encoding="utf-8")

# -----------------------------------------------------

def choose_embeddings(model_name, language):
   
    embeddings_dict = {
        "spa": {
            "word2vec": spa_w2v,
            "fasttext": spa_ft,
        },
        "pol": {
            "word2vec": pol_w2v,
            "fasttext": pol_ft,
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
        print('\nLoading Multilingual BERT embeddings...')
        # load BERT embeddings
        tokenizer = transformers.AutoTokenizer.from_pretrained(mult_bert)
        model = transformers.AutoModel.from_pretrained(mult_bert)
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
    not_found = 0 # for Word2Vec embeddings
    for _, row in data.iterrows():
        pivot, inflection = row["pivot"], row["inflection"]

        if model_name == "FastText":
            vec_pivot = model.get_word_vector(pivot) # vector of the pivot
            vec_inflection = model.get_word_vector(inflection) # vector of the inflection

        elif model_name == "Word2Vec":
            if pivot in model and inflection in model: # need to check since w2v does not use subwords
                vec_pivot = model[pivot]
                vec_inflection = model[inflection]  # Vector of the inflection
            else:
                not_found += 1
                continue

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
    results_df.to_csv(f"py/results/{language}/{language}_{model_name.lower()}_results.csv", index=False, header=["pivot", "inflection", "P-I similarity"])
    print("Results by row saved!")

    # similarities for mean
    sim_inflection_values = [r[2] for r in results]
    print(f"{model_name.upper()} EMBEDDINGS IN {language.upper()}")
    print(f"MEAN SIMILARITY (VERB-INFLECTION): {np.mean(sim_inflection_values):2f}")

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

model, model_name, tokenizer, language = choose_embeddings("word2vec", "pol") # POLISH
calculate_sims(model, model_name, tokenizer, language, um_pol)