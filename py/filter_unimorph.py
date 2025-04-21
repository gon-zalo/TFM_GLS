# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import pandas as pd

# UNFILTERED INFLECTION DATASETS
spa_inf = "datasets/spa/spa.txt"
pol_inf = "datasets/pol/pol.txt"

def filter_inflections(data):
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

# filter_inflections(spa_inf)
# filter_inflections(pol_inf)