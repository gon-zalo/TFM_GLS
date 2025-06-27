# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd


def annotate_pair_type(row):
    verb = row['verb']
    pair = row['pair']

    if pair[:2] == verb[:2]:
        return "suffixed"
    if pair[-3:] != verb[-3:]:
        return "suppletion"
    if pair[:2] != verb[:2]:
        return "empty"
    
# df["pair_type"] = df.apply(annotate_pair_type, axis=1)


df_sgjp = pd.read_csv("datasets/pol/verbs_sgjp.txt", sep="\t") # verb file

impfs = df_sgjp['verb'] # list of imperfectives to pass into the scraping script

# MAIN LOOP
def get_derived_from_list(list): 
    
    '''USE WITH A LIST! Initial function for testing purposes, takes one list of words (verbs) as an argument. Returns a dataframe.'''

    loop = 0
    scraped_verbs = {}
    for imperfective in impfs:

        loop += 1
        print(f"\nImperfective #{loop}: {imperfective}", end="\r")
        
        response = requests.get(f"https://pl.wiktionary.org/wiki/{imperfective}")
        soup = BeautifulSoup(response.text, 'html.parser')

        spans = soup.find_all('span', {'data-mw': True}) # all spans that contain data-mw
        pokrewne_span = "" # empty string to reset it to every loop so that forms that do not have any perfectives do not reuse the perfectives of the previous verb

        for span in spans:
            data_mw = span.get("data-mw") #accessing data-wm
            if data_mw:
                try:
                    data = json.loads(data_mw) # loading it as a json
                    parts = data.get('parts', [])
                    for i, part in enumerate(parts):
                        if isinstance(part, dict): # we need to access a tag which contains "czas" (czasownik/verb) that is a couple levels in (wt inside target inside template)
                            template = part.get('template', {})
                            target = template.get('target', {})
                            if target.get('wt') == 'czas':
                                pokrewne_span = span # we are interested in the span that contains "czas"
                                slice_index = i # index to slice the data afterwards
                                break
                    else:
                        continue
                    break  # if found "czas" then break loop, this way we 
                except json.JSONDecodeError:
                    continue

        if pokrewne_span != "": # if there is a "czas" span, then we process it. Some verbs might not contain related (pokrewne) verbs

            # data processing, finding where the verbs in the span are
            data = json.loads(pokrewne_span['data-mw'])
            data = data['parts']

            data = data[slice_index:] # slicing the span data

            # verbs code
            start_index = 1
            verbs = data[start_index:len(data):2]

            verb_list = [] # empty list to store the verbs
            for verb in verbs:
                if verb == "\n ":
                    break
                else:
                    verb_list.append(verb)

            # aspect tags code
            aspects = data[2:len(data):2]

            aspects = list(aspects)

            aspect_list = []
            for line in aspects:
                found = False
                # print(line)
                if "ndk" in str(line) and found == False:
                    tag = "ndk"
                    found = True
                    aspect_list.append(tag)
                elif "dk" in str(line) and found == False:
                    tag = "dk"
                    found = True
                    aspect_list.append(tag)

            if not aspect_list:
                aspect_list = ['possibly:  dk']
            annotated_verbs = list(zip(verb_list, aspect_list))
            
            perfectives_list = [] # loop to obtain just the perfectives
            
            for verb, aspect in annotated_verbs:
                if aspect == "ndk":
                    pass
                else:
                    verb = verb.strip(" ,[]")
                    verb.strip("]]\n:")
                    if len(verb) < 25 and verb.endswith("ć") or verb.endswith("c") or verb.endswith("się"):
                        perfectives_list.append((verb, aspect))

            if perfectives_list: # assigning the perfectives to the values of the imperfective in the dictionary
                scraped_verbs[imperfective] = perfectives_list
        else:
            print("No pokrewne")
        
    # print(scraped_verbs)
    rows = [] # empty list to store the imperfective and all the perfectives related to it from the dictionary scraped_verbs

    for imperfective, values in scraped_verbs.items(): # unpacking the dictionary
        for perfective, aspect in values:
            rows.append((imperfective, perfective, aspect))

    df = pd.DataFrame(rows, columns= ["pivot", "perfective", "aspect"])

    print("dataframe created")
    print(df.head())

    return df

def get_derived(word): 
    
    '''get derived verbs from wikislownik, USE WITH A PREVIOUSLY MADE DATAFRAME. Takes a word from a column as argument. Returns a list of derived verbs to be appended later on to the initial dataframe. It is a rough script which introduces formatting errors that need to be manually checked, but works well enough'''
    
    print(f"\nImperfective: {word}")
    
    response = requests.get(f"https://pl.wiktionary.org/wiki/{word}")
    soup = BeautifulSoup(response.text, 'html.parser')

    spans = soup.find_all('span', {'data-mw': True}) # all spans that contain data-mw
    pokrewne_span = "" # empty string to reset it to every loop so that forms that do not have any perfectives do not reuse the perfectives of the previous verb

    for span in spans:
        data_mw = span.get("data-mw") #accessing data-wm
        if data_mw:
            try:
                data = json.loads(data_mw) # loading it as a json
                parts = data.get('parts', [])
                for i, part in enumerate(parts):
                    if isinstance(part, dict): # we need to access a tag which contains "czas" (czasownik/verb) that is a couple levels in (wt inside target inside template)
                        template = part.get('template', {})
                        target = template.get('target', {})
                        if target.get('wt') == 'czas':
                            pokrewne_span = span # we are interested in the span that contains "czas"
                            slice_index = i # index to slice the data afterwards
                            break
                else:
                    continue
                break  # if found "czas" then break loop, this way we 
            except json.JSONDecodeError:
                continue

    if pokrewne_span != "": # if there is a "czas" span, then we process it. Some verbs might not contain related (pokrewne) verbs

        # data processing, finding where the verbs in the span are
        data = json.loads(pokrewne_span['data-mw'])
        data = data['parts']

        data = data[slice_index:] # slicing the span data

        # verbs code
        start_index = 1
        verbs = data[start_index:len(data):2]

        verb_list = [] # empty list to store the verbs
        for verb in verbs:
            if verb == "\n ":
                break
            else:
                verb_list.append(verb)

        # aspect tags code
        aspects = data[2:len(data):2]

        aspects = list(aspects)

        aspect_list = []
        for line in aspects:
            found = False
            # print(line)
            if "ndk" in str(line) and found == False:
                tag = "ndk"
                found = True
                aspect_list.append(tag)
            elif "dk" in str(line) and found == False:
                tag = "dk"
                found = True
                aspect_list.append(tag)

        if not aspect_list:
            aspect_list = ['possibly:  dk']
        annotated_verbs = list(zip(verb_list, aspect_list))
        
        perfectives_list = [] # loop to obtain just the perfectives
        
        for verb, aspect in annotated_verbs:
            if aspect == "ndk":
                pass
            else:
                verb = verb.strip(" ,[]")
                verb.strip("]]\n:")
                if len(verb) < 25 and verb.endswith("ć") or verb.endswith("c") or verb.endswith("się"):
                    perfectives_list.append((verb, aspect))

        if perfectives_list: # assigning the perfectives to the values of the imperfective in the dictionary
            return perfectives_list
    else:
        print("No pokrewne")

# the code is appending rows to the dataframe that is being passed to get_derived, for testing purposes use get_derived_from_list
rows = [] # empty list to store the rows to be appended to the dataframe

for _,row in df_sgjp.iterrows():
    rows.append(row)
    verb = row["verb"]
    aspect = row["aspect"]
    id = row["id"]
    pair_empty = row["pair"]
    
    derived_verbs = get_derived(row["verb"])
    if derived_verbs:
        for derived, derived_aspect in derived_verbs:

            pair = derived
            pair_aspect = derived_aspect

            if pair_empty == pair:
                pass

            else:
                new_row = {"verb": verb, "aspect":aspect, "id":id, "pair" : pair, "pair_aspect": pair_aspect, "pair_type" : "derived"}

                rows.append(pd.Series(new_row))
        

print("dataframe created")

expanded_df = pd.DataFrame(rows).reset_index(drop=True)
print()
print(expanded_df.head())

# expanded_df.to_csv("datasets/pol/verbs_sgjp_expanded.txt", sep= "\t", index=False, columns=["verb", "aspect", "id", "pair", "pair_aspect", "pair_id", "pair_type"])

print("\ndf saved")