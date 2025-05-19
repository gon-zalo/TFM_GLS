# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import pandas as pd

# UNFILTERED INFLECTION DATASETS
spa_inf = "datasets/spa/spa.txt"
pol_inf = "datasets/pol/pol.txt"

# UNFILTERED DERIVATION DATASETS

spa_der = pd.read_csv("datasets/spa/spa_derivations.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])
pol_der = pd.read_csv("datasets/pol/pol_derivations.txt", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])

def clean_inflection(data):
    if data == spa_inf:
        language = "spa"
        print("Filtering Spanish data...")
    elif data == pol_inf:
        language = "pol"
        print("Filtering Polish data...")

    df = pd.read_csv(data, sep="\t", header=None, names=["pivot", "inflection", "category"])
    # filtering data to only obtain presente, pret. impf. and futuro simple
    if language == "spa":
        df = df[
            df["category"].str.contains("V;IND;PRS") | # presente
            df["category"].str.contains("V;IND;PST;IPFV") | # pret. impf.
            df["category"].str.contains("V;IND;FUT") # futuro simple
        ]

        # removing vos forms
        df = df[~((df['inflection'].str.endswith('ás') | df['inflection'].str.endswith('és') | df['inflection'].str.endswith('ís')) & df['category'].str.contains('V;IND;PRS'))]

        # removing usted forms
        df = df[~df['category'].str.contains('FORM')]

    elif language == "pol":
        df = df[
            df["category"].str.contains("V;PRS") | # czas teraźniejszy
            df["category"].str.contains("V;PST") | # czas przeszły
            df["category"].str.contains("V;FUT") # czas przyszły
        ]

    print(df.describe())

    # save filtered dataframe
    df.to_csv(f"datasets/{language}/{language}_inflections.txt", sep="\t", index=False, header=False)
    print("Filtered data saved!")

def filter_inflection(dataframe, language):

    if language == "spa":
        print("Filtering Spanish data...")
    elif language == "pol":
        print("Filtering Polish data...")

    df = dataframe.copy()

    # filtering data to only obtain presente, pret. impf. and futuro simple
    if language == "spa":
        present_df =  df[df["category"].str.contains("V;IND;PRS", na=False)] # presente
        present_df = present_df.reset_index(drop=True) # reset index after filtering

        past_df = df[df["category"].str.contains("V;IND;PST;IPFV", na=False)] # pret. impf.
        past_df = past_df.reset_index(drop=True)

        future_df = df[df["category"].str.contains("V;IND;FUT", na=False)] # futuro simple
        future_df = future_df.reset_index(drop=True)

    elif language == "pol":
        present_df =  df[df["category"].str.contains("V;PRS", na=False)]
        present_df = present_df.reset_index(drop=True)

        past_df = df[df["category"].str.contains("V;PST", na=False)]
        past_df = past_df.reset_index(drop=True)

        future_df = df[df["category"].str.contains("V;FUT", na=False)]
        future_df = future_df.reset_index(drop=True)

    return present_df, past_df, future_df

def filter_derivation(dataframe, language):

    if language == "spa":
        print("Filtering Spanish data...")
    elif language == "pol":
        print("Filtering Polish data...")

    df = dataframe.copy()

    # unique categories in the dataset
    categories = df['category'].unique().tolist()
    # empty dictionary to store filtered dataframes
    filtered_dfs = {}

    for category in categories:
        if category == "ADV:V": # removing this since there are only 5 instances of it in Spanish UniMorph
            pass

        else:
            # filter the dataframe for each unique category
            filtered_df = df[df['category'] == category].reset_index(drop=True)
            # append the filtered dataframe to the dictionary
            filtered_df.columns = ['pivot', 'derivation', 'category', 'affix']
            filtered_dfs[category] = filtered_df

    # return a dictionary of 16 filtered dataframes
    return filtered_dfs

# filter_derivation(spa_der, 'spa')
# filter_derivation(pol_der, 'pol')

# clean_inflection(spa_inf)
# clean_inflection(pol_inf)