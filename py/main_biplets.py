# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models import  KeyedVectors
from filter_unimorph import filter_derivation, filter_inflection
import requests
import gzip
import shutil

# -----------------------------------------------------

# EMBEDDINGS/MODELS
    # FASTTEXT
spa_ft = "embeddings/spa/cc.es.300.bin" # spanish
pol_ft = "embeddings/pol/cc.pl.300.bin" # polish

    # WORD2VEC
spa_w2v = "embeddings/spa/sbw_vectors.bin" # spanish (SBW)
pol_w2v = "embeddings/pol/nkjp+wiki-forms-all-300-skipg-ns.bin" # polish

# DATASETS
    # INFLECTION
spa_inf = pd.read_csv("datasets/spa/spa_inflections.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])
pol_inf = pd.read_csv("datasets/pol/pol_inflections.txt", sep="\t", header=None, names=["pivot", "inflection", "category"])

    # DERIVATION
spa_der = pd.read_csv("datasets/spa/spa_derivations.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])
pol_der = pd.read_csv("datasets/pol/pol_derivations.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])

# POLISH ASPECT DATAFRAME
aspect_df = pd.read_csv("datasets/pol/verbs_sgjp_with_derived.txt", sep="\t")

# data for to download and extract word2vec embeddings
sbw_embeddings_url = 'https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz'
sbw_gz_file = 'embeddings/spa/SBW-vectors-300-min5.bin.gz'
sbw_embeddings = "embeddings/spa/sbw_vectors.bin"

ipipan_embeddings_url = 'http://dsmodels.nlp.ipipan.waw.pl/dsmodels/nkjp+wiki-forms-all-300-skipg-ns.txt.gz'
ipipan_gz_file = 'embeddings/pol/nkjp+wiki-forms-all-300-skipg-ns.txt.gz'
ipipan_embeddings = "embeddings/pol/nkjp+wiki-forms-all-300-skipg-ns.txt"

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

    #  WORD2VEC
    if model_name.lower() == "word2vec":

        if language == "Polish":
            # check if embeddings file exists
            if not os.path.exists(pol_w2v):
                print(f"{pol_w2v} not found. Downloading, extracting and converting to .bin...")

                # download .gz file if it doesn't exist
                if not os.path.exists(ipipan_gz_file):
                    print(f"Downloading {ipipan_gz_file} from {ipipan_embeddings_url}...")
                    response = requests.get(ipipan_embeddings_url, stream=True)
                    with open(ipipan_gz_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"Download of {ipipan_gz_file} completed.")

                print(f"Extracting {ipipan_gz_file} to {ipipan_embeddings}...")
                with gzip.open(ipipan_gz_file, 'rb') as f_in:
                    with open(ipipan_embeddings, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Extraction completed, saved as {ipipan_embeddings}.")

                print("Converting Polish embeddings to .bin")
                # converted .txt embeddings file into .bin for faster performance
                model_to_convert = KeyedVectors.load_word2vec_format(ipipan_embeddings, binary=False)
                model_to_convert.save_word2vec_format("embeddings/pol/nkjp+wiki-forms-all-300-skipg-ns.bin", binary=True)
                print("Polish embeddings succesfully converted to .bin!")
                os.remove(ipipan_embeddings)
                os.remove(ipipan_gz_file)
                print("Polish .txt file embeddings and .gz file removed!")

            else:
                print(f"{pol_w2v} found!")

        if language == "Spanish":
            if not os.path.exists(spa_w2v):
                print(f"{spa_w2v} not found. Downloading and extracting...")

                # download .gz file if it doesn't exist
                if not os.path.exists(sbw_gz_file):
                    print(f"Downloading {sbw_gz_file} from {sbw_embeddings_url}...")
                    response = requests.get(sbw_embeddings_url, stream=True)
                    with open(sbw_gz_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"Download of {sbw_gz_file} completed.")
            
                print(f"Extracting {sbw_gz_file} to {spa_w2v}...")
                with gzip.open(sbw_gz_file, 'rb') as f_in:
                    with open(spa_w2v, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Extraction completed, saved as {spa_w2v}.")
                os.remove(sbw_gz_file)
                print("Spanish .gz file removed!")

            else:
                print(f"{spa_w2v} found!")

        print(f'\nLoading {language} Word2Vec embeddings...')
        # load the w2v embeddings
        model = KeyedVectors.load_word2vec_format(embeddings, binary=True)
        return model, "Word2Vec", language

    # FASTTEXT
    if model_name.lower() == "fasttext":
        if language == "Spanish":
        
            if not os.path.exists(spa_ft):
                print("Spanish FastText embeddings not found!")
                print(f"Downloading {language} FastText embeddings...")
                print("Embeddings need to be extracted once they download")
                fasttext.util.download_model('es')
                os.rename("cc.es.300.bin", spa_ft)
                os.remove("cc.es.300.bin.gz")
                print("Embeddings downloaded!")   

        if language == "Polish":

            if not os.path.exists(pol_ft):
                print("Polish FastText embeddings not found!")
                print(f"Downloading {language} FastText embeddings...")
                print("Embeddings need to be extracted once they download")
                fasttext.util.download_model('pl')
                os.rename("cc.pl.300.bin", pol_ft)
                os.remove("cc.pl.300.bin.gz")
                print("Embeddings downloaded!")                

        print(f'\nLoading {language} FastText embeddings...')            
        model = fasttext.load_model(embeddings)
        return model, "FastText", language

    else:
        raise ValueError("Invalid model name.")

def sim_aspect(model, model_name, language, data):

    if language == "Polish":
        language = "pol"

    # Calculate cosine similarities between pivot and inflection
    print("Calculating similarities...")
    results = []

    not_found = 0 # for Word2Vec embeddings
    for _, row in data.iterrows():
        verb, aspect, pair, pair_aspect, pair_type = row["verb"], row["aspect"], row["pair"], row["pair_aspect"], row["pair_type"]

        if model_name == "FastText":
            # print("Pivot", verb)
            verb_embedding = model.get_word_vector(verb) # vector of the verb
            # print("Perfective", pair)
            pair_embedding = model.get_word_vector(pair) # vector of the pair

        elif model_name == "Word2Vec":
            if verb in model and pair in model: # need to check since w2v does not use subwords
                verb_embedding = model[verb]
                pair_embedding = model[pair]  # Vector of the pair
            else:
                not_found += 1
                continue

        # calculate similarity between verb and pair
        similarity = 1 - cosine(verb_embedding, pair_embedding) # need 1 - cosine because cosine alone just measures distance, not similarity

        # append everything to the results list
        results.append((verb, aspect, pair, pair_aspect, pair_type, similarity))

    if model_name == "Word2Vec":
        print(f"Number of words not found in Word2Vec model: {not_found}")

    # create a dataFrame with the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/{language}/{language}_{model_name.lower()}_aspect_results.csv", index=False, header=["verb", "aspect", "pair", "pair_aspect", "pair_type", "similarity"])
    print("Results by row saved!")

    # # calculate and print the mean similarity
    # similarity_values = [r[2] for r in results]
    # print(f"\n{model_name.upper()} EMBEDDINGS IN {language.upper()}")
    # print(f"    MEAN SIMILARITY (PIVOT-PERFECTIVE): {np.mean(similarity_values):.2f}")

def sim_inflection(model, model_name, language, data):
    if language == "Spanish":
        language = "spa"
    elif language == "Polish":
        language = "pol"

    # Calculate cosine similarities between pivot and inflection
    print("Calculating similarities...")
    results = []

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

def shuffle_inflection(model, model_name, language, data, tense):
    # function to shuffle inflections and extract the mean similarity of each loop
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
    
        data[["inflection", "category"]] = data[["inflection", "category"]].sample(frac=1, random_state=123 + number).reset_index(drop=True)

        # this empties both lists at the start of each loop
        results = [] # empty list to store all the columns of the df
        similarity_values = [] # empty list to store the similarity values of the whole dataframe

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

def sim_derivation(model, model_name, language, data):
    if language == "Spanish":
        language = "spa"
    elif language == "Polish":
        language = "pol"

    # calculate cosine similarities between pivot and derivation
    print("Calculating similarities...")
    results = []

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

def shuffle_derivation(model, model_name, language, data, name):
     # same as shuffle_inflection but for derivational data
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
    
        data[["derivation", "category", "affix"]] = data[["derivation", "category", "affix"]].sample(frac=1, random_state=123 + number).reset_index(drop=True)

        # this empties both lists at the start of each loop
        results = [] # empty list to store all the columns of the df
        similarity_values = [] # empty list to store the similarity values of the whole dataframe

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
EMBEDDINGS WILL DOWNLOAD AUTOMATICALLY IF THEY ARE NOT IN THE DIRECTORY
choose_embeddings takes an argument of the model name (fasttext, word2vec or bert) and a language argument (spa or pol), it outputs the model, the model name and the language (to be used in the results file name and file path).

calculate_sims takes what choose_embeddings outputs and an argument of the file to be used.
'''

###### INFLECTION ######
# # FASTTEXT
# model, model_name, language = choose_embeddings("fasttext", language="spa") # SPANISH
# sim_inflection(model, model_name, language, data=spa_inf_shuf)

# model, model_name, language = choose_embeddings("fasttext", "pol") # POLISH
# sim_inflection(model, model_name, language, pol_inf_shuf)

# # WORD2VEC
# model, model_name, language = choose_embeddings("word2vec", "spa") # SPANISH
# sim_inflection(model, model_name, language, spa_inf_shuf)

# model, model_name, language = choose_embeddings("word2vec", "pol") # POLISH
# sim_inflection(model, model_name, language, pol_inf_shuf)

# ###### DERIVATION ######
# # FASTTEXT
# model, model_name, language = choose_embeddings("fasttext", "spa")
# sim_derivation(model, model_name, language, spa_der_shuf)

# model, model_name, language = choose_embeddings("fasttext", "pol")
# sim_derivation(model, model_name, language, pol_der_shuf)

# # WORD2VEC
# model, model_name, language = choose_embeddings("word2vec", "spa")
# sim_derivation(model, model_name, language, spa_der_shuf)

# model, model_name, language = choose_embeddings("word2vec", "pol")
# sim_derivation(model, model_name, language, pol_der_shuf)

####### BASELINE FUNCTIONS #######

# DERIVATION
# spanish
# filtered_dfs = filter_derivation(spa_der, 'spa')
# model, model_name, language = choose_embeddings("fasttext", "spa")
# for name, dataframe in filtered_dfs.items():
#     shuffle_derivation(model, model_name, language, dataframe, name)

# model, model_name, language = choose_embeddings("word2vec", "spa")
# for name, dataframe in filtered_dfs.items():
#     shuffle_derivation(model, model_name, language, dataframe, name)

# # polish
# filtered_dfs = filter_derivation(pol_der, 'pol')
# model, model_name, language = choose_embeddings("fasttext", "pol")
# for name, dataframe in filtered_dfs.items():
#     shuffle_derivation(model, model_name, language, dataframe, name)

# model, model_name, language = choose_embeddings("word2vec", "pol")
# for name, dataframe in filtered_dfs.items():
#     shuffle_derivation(model, model_name, language, dataframe, name)


# # INFLECTION
# # spanish
# present_df, past_df, future_df = filter_inflection(spa_inf, 'spa')
# model, model_name, language = choose_embeddings("fasttext", "spa")
# shuffle_inflection(model, model_name, language, present_df, "present")
# shuffle_inflection(model, model_name, language, past_df, 'past')
# shuffle_inflection(model, model_name, language, future_df, 'future')

# model, model_name, language = choose_embeddings("word2vec", "spa")
# shuffle_inflection(model, model_name, language, present_df, "present")
# shuffle_inflection(model, model_name, language, past_df, 'past')
# shuffle_inflection(model, model_name, language, future_df, 'future')

# # polish
# present_df, past_df, future_df = filter_inflection(pol_inf, 'pol')
# model, model_name, language = choose_embeddings("fasttext", "pol")
# shuffle_inflection(model, model_name, language, present_df, "present")
# shuffle_inflection(model, model_name, language, past_df, 'past')
# shuffle_inflection(model, model_name, language, future_df, 'future')

# model, model_name, language = choose_embeddings("word2vec", "pol")
# shuffle_inflection(model, model_name, language, present_df, "present")
# shuffle_inflection(model, model_name, language, past_df, 'past')
# shuffle_inflection(model, model_name, language, future_df, 'future')

# # ASPECT STUDY FUNCTIONS
# model, model_name, language = choose_embeddings("fasttext", "pol") # POLISH
# sim_aspect(model, model_name, language, aspect_df)

# model, model_name, language = choose_embeddings("word2vec", "pol") # POLISH
# sim_aspect(model, model_name, language, aspect_df)