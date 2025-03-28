import pandas as pd

esp_data = "py/datasets/spa.txt"
pol_data = "py/datasets/pol.txt"

def filter_unimorph(data):
    if data == esp_data:
        name = "spa"
        print("Filtering Spanish data...")
    elif data == pol_data:
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
            df["category"].str.contains("V;PRS") | # present (czas teraźniejszy)
            df["category"].str.contains("V;PST") | # pret. impf.
            df["category"].str.contains("V;FUT") # futuro simple
        ]

    print(filtered_df.describe())

    # save filtered dataframe
    filtered_df.to_csv(f"py/datasets/filtered_{name}.txt", sep="\t", index=False, header=False)
    print("Filtered data saved!")

# running the function for spanish and polish data
filter_unimorph(esp_data)
filter_unimorph(pol_data)