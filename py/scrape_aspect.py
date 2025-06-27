# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import tqdm

'''
CODE TO SCRAPE ASPECTUAL PAIRS FROM SGJP
'''


json_file = "datasets/pol/sgjp_frequent_8k.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)


extracted = []

for item in data:
    extracted.append((item["entry"], item["genders"], item["id"]))

imperfective_verbs = []

for (verb, aspect, id) in extracted:
    if aspect == "ndk" and (verb.endswith("ć") or verb.endswith("c")):
        imperfective_verbs.append((verb, aspect, id))
    else:
        pass
        

print("Total imperfective verbs:", len(imperfective_verbs))

verb_data = []


for (verb, aspect, id) in tqdm.tqdm(imperfective_verbs):
    print(f"Fetching pair(s) of... {verb}")

    odpowiedniki = [] # list in case there are more than 1 empty prefixed verb
    url = f"http://sgjp.pl/edycja/ajax/inflection-tables/?lexeme_id={id}&variant=2"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    trs = soup.find_all("tr")

    if len(trs) > 1:
        tr = trs[1]
        
        if tr:
            spans = tr.find_all("span")

            if spans:
                for span in spans:

                    odpowiednik = span.string
                    id_tag = span.get("id")

                    if odpowiednik and id_tag:

                        if odpowiednik.endswith("ć") or odpowiednik.endswith("c"):

                            odpowiednik_id = "".join([character for character in id_tag if character.isdigit()])
                            odpowiednik_id = int(odpowiednik_id)

                            odpowiedniki.append(odpowiednik)
        else:
            pass

    if aspect == "ndk":
        odpowiednik_aspect = "dk"
    elif aspect == "dk":
        odpowiednik_aspect = "ndk"

    # pair.append((odpowiednik, odpowiednik_id))
    if len(odpowiedniki) > 1:
        for odpowiednik in odpowiedniki:
            verb_data.append((verb, aspect, id, odpowiednik, odpowiednik_aspect, odpowiednik_id))
    else:
        verb_data.append((verb, aspect, id, odpowiednik, odpowiednik_aspect, odpowiednik_id))

df = pd.DataFrame(verb_data, columns=["verb", "aspect", "id", "pair", "pair_aspect", "pair_id"])


# df.to_csv("datasets/pol/verbs_sgjp_fixed.txt", sep="\t", index=False, columns=["verb", "aspect", "id", "pair", "pair_aspect", "pair_id"])