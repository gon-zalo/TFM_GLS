# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import pandas as pd

# UNFILTERED INFLECTION DATASETS
spa_inf = "datasets/spa/spa.txt"
pol_inf = "datasets/pol/pol.txt"

# DERIVATION
um_spa_der = pd.read_csv("datasets/spa/spa.derivations", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])
um_pol_der = pd.read_csv("datasets/pol/pol.derivations", sep="\t", header=None, names=["pivot", "derivation", "category", "affix"])

def filter_inflections(data):
    if data == spa_inf:
        name = "spa"
        print("Filtering Spanish data...")
    elif data == pol_inf:
        name = "pol"
        print("Filtering Polish data...")

    df = pd.read_csv(data, sep="\t", header=None, names=["pivot", "inflection", "category"])
    # filtering data to only obtain presente, pret. impf. and futuro simple
    if name == "spa":
        filtered_df = df[
            df["category"].str.contains("V;IND;PRS") | # presente
            df["category"].str.contains("V;IND;PST;IPFV") | # pret. impf.
            df["category"].str.contains("V;IND;FUT") # futuro simple
        ]
    elif name == "pol":
        filtered_df = df[
            df["category"].str.contains("V;PRS") | # czas teraźniejszy
            df["category"].str.contains("V;PST") | # czas przeszły
            df["category"].str.contains("V;FUT") # czas przyszły
        ]

    print(filtered_df.describe())

    # save filtered dataframe
    filtered_df.to_csv(f"py/datasets/filtered_{name}.txt", sep="\t", index=False, header=False)
    print("Filtered data saved!")


# def process_derivations(data):



# print(um_spa_der[um_spa_der["category"].str.endswith(":U")])

# filter_inflections(spa_inf)
# filter_inflections(pol_inf)