# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import pandas as pd
pd.set_option('display.max_rows', 20)

spa_der = pd.read_csv("datasets/spa/spa_derivations.txt", header=None, names=['pivot', 'derivation', 'category', 'affix'], sep='\t')

pol_der = pd.read_csv('datasets/pol/pol_derivations.txt', header=None, names=['pivot', 'derivation', 'category', 'affix'], sep='\t')


# print(spa_der['affix'].value_counts().to_string())

# print(pol_der['affix'].value_counts().to_string())