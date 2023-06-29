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

df = pd.read_csv("./output_files/results/helpdesk/suffix_and_remaining_time_helpdesk.csv")
df = df.rename(columns={"Prefix length": "Prefix"})

# Formula:
# Ynew = * c * row["Predicted times"]) + (Ymed * (1 - c))
# sum all of the Ynew - row["Ground truth"] and see which one is the best

## Theoretisch müsste doch mit einer größeren Prefix Länge unser c auch steigen => Weil wir haben einen größeren Faktor von shared

averageMaeNew = []
averageMaeOld = []

caseID = df.loc[df['Prefix'] == 4]["CaseID"]


## Only get Entries which did full length e.g until 12 or until 4
df = df[df['CaseID'].isin(caseID)]


## Check if this is the right data
for k in range(2, 5):
    dfHelp = df.loc[df['Prefix'] == k]
    print("Average MAE")
    averageMAE = sum(dfHelp["MAE"]) / len(dfHelp)
    print(averageMAE / (60*60*24))

## Get test CaseIDs
train, test = train_test_split(df.loc[df['Prefix'] == 2], test_size=0.2)
rightIDs = train["CaseID"]
dfTrain = df[df['CaseID'].isin(rightIDs)]
dfTest = df[~df['CaseID'].isin(rightIDs)]


## Problem why the means are so different: Some Values are extracted which have a high mean

## Lets start for Prefix = 2
for k in range(2, 5):
    train = dfTrain
    test = dfTest

    print("Length of dataset")
    print(len(train))
    print(len(test))
    
    
    #dfNew['Predicted times'] = dfNew['Predicted times'].str.replace('.', '').astype(int)
    #dfNew['MAE'] = dfNew['MAE'].str.replace('.', '').astype(int)
    #dfNew = df
    Ymed = train["Predicted times"].mean()
    train["Ynew"] = None
    train["MAE_New"] = None
    print(Ymed)
    overAllSum = 100000000000
    MaeAndC = []
    best_confidence = 0

    print(k)
    print(train[["MAE", "CaseID"]])




    ###Why is my mean always pretty small, when I include many values and large when I include not many values?


    ## Problem: Sehr hohe Variabilität in den Daten
    print("Here comes Variance")
    variance = train['MAE'].var()

    print(variance)
    print("Variance in hours")
    print(variance / (60 * 60 * 24))

    ## So the Variance is pretty high in my dataset

    #test.plot(x='CaseID', y='MAE', style='o')
    #plt.show()
    ##Plot MAE vs ID
    
    
    

    print("Here comes different mean")
    print(train["MAE"].mean() / (60 * 60 * 24))
    print(test["MAE"].mean() / (60 * 60 * 24))
    
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

    
    
    
    x = [d['confidence'] for d in MaeAndC]
    y = [d['MAE'] for d in MaeAndC]

    plt.clf()
    plt.plot(x, y, marker='o')
    plt.xlabel('Confidence')
    plt.ylabel('MAE')
    plt.title('Confidence vs. MAE')
    plt.grid(True)

    plt.savefig("Prefix" + str(k) + ".png")
    
    test["Ynew"] = None
    test["MAE_New"] = None
    YmedTest = test["Predicted times"].mean()
    for index, row in test.iterrows():
        row["Ynew"] = (best_confidence * row["Predicted times"]) + (YmedTest * (1 - best_confidence))
        test.loc[index, "Ynew"] = row["Ynew"]
        test.loc[index, "MAE_New"] = abs(row["Ynew"] - row["Ground truth times"])


    averageMaeNew.append((test["MAE_New"].mean()) / (60 * 60 * 24))
    averageMaeOld.append((test["MAE"].mean()) / (60 * 60 * 24))

    print(k)
    print(test[["MAE", "CaseID"]])
    
    labels = ["MAE average no adjustment", "MAE average with adjustment"]
    values = [test["MAE"].mean() / (60 * 60 * 24), test["MAE_New"].mean() / (60 * 60 * 24)]

    plt.clf()
    #Plotting the bar graph
    plt.bar(labels, values)

    #Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Average MAE')
    plt.title('Bar Graph: Value 1 vs. Value 2')

    plt.savefig("Comparision MAE Prefix" + str(k) + ".png")


averageMaeNewValue = sum(averageMaeNew) / len(averageMaeNew)
averageMaeOldValue = sum(averageMaeOld) / len(averageMaeOld)


print("Final Result")
print(averageMaeNewValue)
print(averageMaeOldValue)

print(averageMaeNewValue / averageMaeOldValue)



# Question: Why is the average MAE so different?
    


## Reduction of around 1.5%

## Why isnt it better for different prefix?
## I think the varying in the prefix in the different prefix is due to not enough data
