import re
import pandas as pd

df = pd.read_csv("py/derinet/spa/spanish-wfn.tsv", sep="\t", header=None)

df.columns = ["index", "word1", "word2", "reference", "4", "5"]

df = df[df["word1"].str.match(r"^[a-z]")]

df.to_csv("py/derinet/spa/filtered_spanish-wfn.csv", index=False)
print("Saved")