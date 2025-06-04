# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import tqdm

json_file = "datasets/pol/sgjp_frequent_8k.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)


extracted = []

for item in data:
    extracted.append((item["entry"], item["genders"], item["id"]))

clear_verbs = []

for (verb, aspect, id) in extracted:
    if aspect == "dk" or aspect == "ndk" and verb.endswith("ć") or verb.endswith("c"):
        clear_verbs.append((verb, aspect, id))
    else:
        pass
        

print(f"Removed {len(extracted)-len(clear_verbs)} biaspectual verbs")
print("Total verbs:", len(clear_verbs))

verb_data = []


for (verb, aspect, id) in tqdm.tqdm(clear_verbs):
    print(f"Fetching pair of... {verb}")

    url = f"http://sgjp.pl/edycja/ajax/inflection-tables/?lexeme_id={id}&variant=2"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    trs = soup.find_all("tr")

    if len(trs) > 1:
        tr = trs[1]
        
        if tr:
            span = tr.find("span")

            if span:
                odpowiednik = span.string
                id_tag = span.get("id")

                if odpowiednik and id_tag:
                    odpowiednik_id = "".join([character for character in id_tag if character.isdigit()])
                    odpowiednik_id = int(odpowiednik_id)
        else:
            pass

    if aspect == "ndk":
        odpowiednik_aspect = "dk"
    elif aspect == "dk":
        odpowiednik_aspect = "ndk"

    # pair.append((odpowiednik, odpowiednik_id))
    if odpowiednik:
        verb_data.append((verb, aspect, id, odpowiednik, odpowiednik_aspect, odpowiednik_id))


df = pd.DataFrame(verb_data, columns=["verb", "aspect", "id", "pair", "pair_aspect", "pair_id"])

df.to_csv("datasets/pol/verbs_sgjp.txt", sep="\t", index=False, columns=["verb", "aspect", "id", "pair", "pair_aspect", "pair_id"])