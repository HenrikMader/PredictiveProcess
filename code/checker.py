'''
Evaluate the output of LSTM Model

Author: Henrik Mader
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.optimize as spo

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv("./output_files/results/bpi_12/suffix_and_remaining_time_permitwbpi_12_w.csv")
df = df.rename(columns={"Prefix length": "Prefix"})


dfHelp = df.loc[df['Prefix'] == 2]
print("Average MAE")
averageMAE = sum(dfHelp["MAE"]) / len(dfHelp)
print(averageMAE / (60*60*24))
