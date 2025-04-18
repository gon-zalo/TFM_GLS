# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import pandas as pd
import json

# data
spa_inf = pd.read_csv("datasets/spa/spa_inflections.txt", header=None, names=['pivot', 'inflection', 'category'], sep='\t')
crea_data = pd.read_csv("datasets/spa/lemas_crea_10k.txt", sep="\t", on_bad_lines='skip', skiprows=3, names=['lemma', 'category', 'freq1', 'freq2', 'freq3'], )

pol_inf = pd.read_csv('datasets/pol/pol_inflections.txt', header=None, names=['pivot', 'inflection', 'category'], sep='\t')
sgjp_json = "datasets/pol/sgjp_frequent_8k.json"

# spanish code
spa_pivots = list(set(spa_inf['pivot'])) # unique pivots
spa_lemmas = list(set(crea_data['lemma']))

spa_most_freq = []
for lemma in spa_lemmas:
    if lemma in spa_pivots:
        spa_most_freq.append(lemma)

subset_spa = spa_inf[spa_inf['pivot'].isin(spa_most_freq)]

# polish code
# Load the JSON data from a file
with open(sgjp_json, "r", encoding="utf-8") as file:
    data = json.load(file)  # Load the JSON data into a Python list of dictionaries

pol_pivots = list(set(pol_inf['pivot']))
pol_lemmas = []
# appends the [entry] key (the verb itself) from the json to pol_lemmas
for verb in data:
    pol_lemmas.append(verb['entry'])

pol_most_freq = []
for lemma in pol_lemmas:
    if lemma in pol_pivots:
        pol_most_freq.append(lemma)

subset_pol = pol_inf[pol_inf['pivot'].isin(pol_most_freq)]

# subset_pol.to_csv('datasets/pol/pol_inflections_subset.txt', index=False, header=None)
# subset_spa.to_csv('datasets/spa/spa_inflections_subset.txt', index=False, header=None)