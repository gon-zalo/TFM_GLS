import os
import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
import transformers
import torch
# -----------------------------------------------------

# EMBEDDINGS/MODELS
ft_embeddings = "py/embeddings/cc.es.300.bin"
word2vec_embeddings = "py/embeddings/sbw_vectors.bin"
bert_embeddings = "bert-base-multilingual-cased"

# DATASETS
triplets_data = pd.read_csv("py/datasets/50_triplets.csv", sep=",") # small data for testing
spa_VV = pd.read_csv("py/datasets/spa.txt", sep="\t", header=None, names=["verb", "inflection", "category"])

# -----------------------------------------------------

chosen_model = False
while chosen_model == False:
    embeddings = input("Choose which embeddings to use: SBW (s) or FastText (f) or Multilingual BERT (b): ")
    if embeddings.lower() in ["sbw", "s"]:
        print('Loading Spanish Billion Words (Word2Vec) embeddings...')
        embeddings = KeyedVectors.load_word2vec_format(word2vec_embeddings, binary=True, limit=50000) # limit is for testing purposes
        emb_name = "SBW" 
        chosen_model=True
    elif embeddings.lower() in ["fasttext", "f"]:
        print('Loading FastText embeddings...')
        if not os.path.exists(ft_embeddings):
            fasttext.util.download_model('es', if_exists='ignore')  # Download Spanish vectors if not available
        embeddings = fasttext.load_model(ft_embeddings)  # loads embeddings
        emb_name = "FastText"
        chosen_model=True
    elif embeddings.lower() in ["bert", "b"]:
        print('Loading BERT embeddings...')
        tokenizer = transformers.AutoTokenizer.from_pretrained(bert_embeddings)
        model = transformers.AutoModel.from_pretrained(bert_embeddings)
        emb_name = "BERT"
        chosen_model = True

        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'
        model.to(device)

    elif embeddings.lower() == "exit":
        exit()
    else:
        print("Invalid choice. Please choose 'SBW', 'fastText' or 'BERT'.")
  

def calculate_sims(embeddings, data):
    # calculate cosine similarities between pivot and inflection/derivation
    print("Calculating similarities...")
    results = []
    for _, row in data.iterrows():
        pivot, inflection, derivation = row["pivot"], row["inflection"], row["derivation"]

        if emb_name == "FastText":
            vec_pivot = embeddings.get_word_vector(pivot) # vector of the pivot
            vec_inflection = embeddings.get_word_vector(inflection) # vector of the inflection
            vec_derivation = embeddings.get_word_vector(derivation) # vector of the derivation

        elif emb_name == "SBW":
            # check if the words are in the model since it does not use subwords
            if pivot in embeddings and inflection in embeddings and derivation in embeddings: # need to check since w2v does not use subwords
                vec_pivot = embeddings[pivot]
                vec_inflection = embeddings[inflection]  # Vector of the inflection
                vec_derivation = embeddings[derivation]  # Vector of the derivation

            else:
                print(f"Word not found: {pivot}, {inflection}, {derivation}")

        elif emb_name == "BERT":
            # Tokenize and encode inputs
            inputs = tokenizer([pivot, inflection, derivation], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use the [CLS] token embedding as the sentence/word representation
            vec_pivot = outputs.last_hidden_state[0][0].numpy()
            vec_inflection = outputs.last_hidden_state[1][0].numpy()
            vec_derivation = outputs.last_hidden_state[2][0].numpy()
        
        # calculate similarity between pivot inflection/derivation
        # need 1 - cosine because cosine alone just measures distance, not similarity
        sim_inflection = 1 - cosine(vec_pivot, vec_inflection)
        sim_derivation = 1 - cosine(vec_pivot, vec_derivation)

        # calculate vector offsets
        offset_inflection = vec_inflection - vec_pivot
        offset_derivation = vec_derivation - vec_pivot

        # append everything to the list
        results.append((pivot, inflection, derivation, sim_inflection, sim_derivation, offset_inflection, offset_derivation))

    # print results in a table format
    print(f"{'Pivot':<15}{'Inflection':<15}{'Derivation':<15}{'P-I similarity':<15}{'P-D similarity':<15}")
    for result in results:
        print(f"{result[0]:<15}{result[1]:<15}{result[2]:<15}{result[3]:<15.2f}{result[4]:<15.2f}")

    # calculates cosine similarity between inflection and derivation offsets and prints it
    similarity = 1 - cosine(offset_inflection, offset_derivation)
    print(f"\n------ RESULTS OF {emb_name.upper()} EMBEDDINGS -----\n")
    print(f"    Cosine similarity between inflection and derivation offsets: {similarity:.2f}")

    # similarities for mean
    sim_inflection_values = [r[3] for r in results]
    sim_derivation_values = [r[4] for r in results]
    print(f"    Inflection mean similarity to pivot: {np.mean(sim_inflection_values):2f}")
    print(f"    Derivation mean similarity to pivot: {np.mean(sim_derivation_values):2f}") 

print(f'{emb_name} embeddings loaded!')

calculate_sims(embeddings, triplets_data)