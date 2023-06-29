'''
Evaluate the output of LSTM Model

Author: Henrik Mader
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.optimize as spo
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv("./output_files/results/env_permit/suffix_and_remaining_time_permitwenv_permit.csv")
df = df.rename(columns={"Prefix length": "Prefix"})


averageMaeNew = []
averageMaeOld = []

allConfidence = []
allMAE = []

caseID = df.loc[df['Prefix'] == 19]["CaseID"]


df = df[df['CaseID'].isin(caseID)]

for i in range(1, 20):


    train, test = train_test_split(df.loc[df['Prefix'] == 2], test_size=0.2)
    rightIDs = train["CaseID"]
    dfTrain = df[df['CaseID'].isin(rightIDs)]
    dfTest = df[~df['CaseID'].isin(rightIDs)]



    bestConfidenceArray = []
    adjustmentMAEArray = []

    
    for k in range(2, 20):
        train = dfTrain.loc[dfTrain['Prefix'] == k]
        test = dfTest.loc[dfTest['Prefix'] == k]
        
        Ymed = train["Predicted times"].mean()
        train["Ynew"] = None
        train["MAE_New"] = None
        overAllSum = 100000000000
        MaeAndC = []
        best_confidence = 0

        
        for i in range(0, 11, 1):
            c = i / 10
            helperSumAbsoluteMae = 0
            for index, row in train.iterrows():
                row["Ynew"] = (c * row["Predicted times"]) + (Ymed * (1 - c))
                row["MAE_New"] = abs(row["Ynew"] - row["Ground truth times"])
                helperSumAbsoluteMae += row["MAE_New"]
            helperSumAverageMAE = helperSumAbsoluteMae / len(train)
            MaeAndC.append({
                "MAE": helperSumAverageMAE / (60 * 60 * 24),
                "confidence": c
                })
            if (helperSumAbsoluteMae < overAllSum):
                overAllSum = helperSumAbsoluteMae
                best_confidence = c

        bestConfidenceArray.append(best_confidence)
        
        test["Ynew"] = None
        test["MAE_New"] = None
        YmedTest = test["Predicted times"].mean()
        for index, row in test.iterrows():
            row["Ynew"] = (best_confidence * row["Predicted times"]) + (YmedTest * (1 - best_confidence))
            test.loc[index, "Ynew"] = row["Ynew"]
            test.loc[index, "MAE_New"] = abs(row["Ynew"] - row["Ground truth times"])


        averageMaeNew.append((test["MAE_New"].mean()) / (60 * 60 * 24))
        averageMaeOld.append((test["MAE"].mean()) / (60 * 60 * 24))

        adjustmentMAEArray.append((1 - (test["MAE_New"].mean() / test["MAE"].mean()))* 100)


    averageMaeNewValue = sum(averageMaeNew) / len(averageMaeNew)
    averageMaeOldValue = sum(averageMaeOld) / len(averageMaeOld)




    ## Investigation for every Prefix
    print("Final Result")
    print(averageMaeNewValue)
    print(averageMaeOldValue)

    print(averageMaeNewValue / averageMaeOldValue)

    print("For plot")
    
    print(bestConfidenceArray)
    print(adjustmentMAEArray)

    Prefix = [i + 2 for i in range(len(bestConfidenceArray))]
    PrefixWhole = Prefix

    #plt.clf()
    #plt.xticks(range(2, 20))
    #plt.plot(PrefixWhole, bestConfidenceArray, 'o')
    #plt.xlabel('Prefix')
    #plt.ylabel('Best confidence')
    #plt.title('Confidence to corresponding Prefix')
    #plt.gca().xaxis.set_major_formatter('{:.0f}'.format)
    #plt.savefig("Prefix and confidence" + ".png")
    #plt.clf()

    #plt.plot(PrefixWhole, adjustmentMAEArray, 'o')
    #plt.xlabel('Prefix')
    #plt.ylabel('Improvement MAE in percent')
    #plt.title('Improvement MAE to corresponding Prefix')
    #plt.gca().xaxis.set_major_formatter('{:.0f}'.format)
    #plt.savefig("Prefix and adjustment"+ ".png")

    allConfidence.append(bestConfidenceArray)
    allMAE.append(adjustmentMAEArray)


## Investigation for average
print(allConfidence)
print(allMAE)

allConfidence = np.array(allConfidence)
allMAE = np.array(allMAE)

averageConfidence = allConfidence.mean(axis=0) 
averageMAEArray = allMAE.mean(axis = 0)


Prefix = [i + 2 for i in range(len(averageConfidence))]
PrefixWhole = [int(prefix) for prefix in Prefix]

plt.clf()
plt.plot(PrefixWhole, averageConfidence, 'o')
plt.xlabel('Prefix')
plt.ylabel('Best correlation')
plt.title('Correlation to corresponding Prefix')
plt.gca().xaxis.set_major_formatter('{:.0f}'.format)
plt.savefig("Prefix and confidence" + ".png")
plt.clf()

plt.plot(PrefixWhole, averageMAEArray, 'o')
plt.xlabel('Prefix')
plt.ylabel('Improvement MAE in percent')
plt.title('Improvement MAE to corresponding Prefix')
plt.gca().xaxis.set_major_formatter('{:.0f}'.format)
plt.savefig("Prefix and adjustment"+ ".png")
