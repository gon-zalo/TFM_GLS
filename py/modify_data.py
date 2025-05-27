# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import pandas as pd
import json

# data
spa_inf = pd.read_csv("datasets/spa/spa_inflections.txt", header=None, names=['pivot', 'inflection', 'category'], sep='\t')
pol_inf = pd.read_csv('datasets/pol/pol_inflections.txt', header=None, names=['pivot', 'inflection', 'category'], sep='\t')
aspect_df = pd.read_csv('datasets/pol/pol', header=None, names=['pivot', 'inflection', 'category', 'aspect'], sep='\t')
pol_inf_an = pd.read_csv('datasets/pol/pol_inflections_an.txt', header=None, names=['pivot', 'inflection', 'category', 'aspect'], sep='\t')

spa_der = pd.read_csv("datasets/spa/spa_derivations.txt", header=None, names=['pivot', 'derivation', 'category', 'affix'], sep='\t')
pol_der = pd.read_csv('datasets/pol/pol_derivations.txt', header=None, names=['pivot', 'derivation', 'category', 'affix'], sep='\t')

    # frequent verbs data
crea_data = pd.read_csv("datasets/spa/lemas_crea_10k.txt", sep="\t", on_bad_lines='skip', skiprows=3, names=['lemma', 'category', 'freq1', 'freq2', 'freq3'])
sgjp_json = "datasets/pol/sgjp_frequent_8k.json" # polish verb list

### SUBSET INFLECTION ###
def subset_inf(data, language):

    pivots = list(set(data['pivot'])) # unique pivots, could also do .value_counts().index

    if language == 'spa':
        lemmas = list(set(crea_data['lemma'])) # lemmas
    else: # for polish

        # load json from sgjp
        with open(sgjp_json, "r", encoding="utf-8") as file:
            verbs_json = json.load(file)
        lemmas = []
        # appends the [entry] key (the verb itself) from the json to pol_lemmas
        for verb in verbs_json:
            lemmas.append(verb['entry'])

    # loop to take only the lemmas that appear in both pivots and lemmas
    most_freq = []
    for lemma in lemmas:
        if lemma in pivots:
            most_freq.append(lemma)

    # subset df
    subset = data[data['pivot'].isin(most_freq)]

    subset.to_csv(f'datasets/{language}/{language}_inflections_subset.txt', index=False, header=None, sep='\t')


### SUBSET DERIVATION ###
def subset_der(data, language):
    if language == 'spa' or 'pol':
        # value_counts outputs each unique value sorted, .index takes the affixes (.values takes counts), transform it into a list and take the first 15 to filter the df
        data_affixes = list(data['affix'].value_counts().index)[:15] # get top 15 affixes

        data_subset = data[data['affix'].isin(data_affixes)] # extract a subset of only those affixes

        # data_subset.to_csv(f'datasets/{language}/{language}_derivations_subset.txt', index=False, header=None, sep='\t') # save


### SHUFFLE DATA ###
def shuffle(data, language):
    if data.columns[1] == 'inflection': # check if the df is inflection or derivation
        for number in range(1,11):
            data[["inflection", "category"]] = data[["inflection", "category"]].sample(frac=1, random_state=123).reset_index(drop=True)
            # data.to_csv(f'datasets/{language}/{language}_inflections_shuffled.txt', index= False, header=None, sep='\t')
    else:
        for number in range(1,11):
            data[["derivation", "category", "affix"]] = data[["derivation", "category", "affix"]].sample(frac=1, random_state=123).reset_index(drop=True)
            # data.to_csv(f'datasets/{language}/{language}_derivations_shuffled.txt', index= False, header=None, sep='\t')        
    2


### CALLING FUNCTIONS
# language should be 'spa' or 'pol'

# subset_der(spa_der, 'spa')
# subset_der(pol_der, 'pol') 

# subset_inf(spa_inf, 'spa')
subset_inf(pol_inf_an, 'pol')

# shuffle(spa_der, 'spa')
# shuffle(spa_inf, 'spa')

# shuffle(pol_der, 'pol')
# shuffle(pol_inf, 'pol')