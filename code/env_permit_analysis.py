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


## Only get Entries which did full length e.g until 12 or until 4
df = df[df['CaseID'].isin(caseID)]

for i in range(1, 20):

    ## Get test CaseIDs
    train, test = train_test_split(df.loc[df['Prefix'] == 2], test_size=0.2)
    rightIDs = train["CaseID"]
    dfTrain = df[df['CaseID'].isin(rightIDs)]
    dfTest = df[~df['CaseID'].isin(rightIDs)]


    ## Problem why the means are so different: Some Values are extracted which have a high mean

    bestConfidenceArray = []
    adjustmentMAEArray = []
    ## Lets start for Prefix = 2
    for k in range(2, 20):
        train = dfTrain.loc[dfTrain['Prefix'] == k]
        test = dfTest.loc[dfTest['Prefix'] == k]

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

        bestConfidenceArray.append(best_confidence)
        
        
        x = [d['confidence'] for d in MaeAndC]
        y = [d['MAE'] for d in MaeAndC]

        plt.clf()
        plt.plot(x, y, marker='o')
        plt.xlabel('Confidence')
        plt.ylabel('MAE')
        plt.title('Confidence vs. MAE')
        plt.grid(True)
        
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


    averageMaeNewValue = sum(averageMaeNew) / len(averageMaeNew)
    averageMaeOldValue = sum(averageMaeOld) / len(averageMaeOld)


    print("Final Result")
    print(averageMaeNewValue)
    print(averageMaeOldValue)

    print(averageMaeNewValue / averageMaeOldValue)

    print("For plot")
    print(bestConfidenceArray)
    print(adjustmentMAEArray)

    ## Plot those things on the y
    ## Prefix on the x

    Prefix = [i + 2 for i in range(len(bestConfidenceArray))]
    PrefixWhole = Prefix

    print(PrefixWhole)

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
    # Question: Why is the average MAE so different?
        


    ## Reduction of around 1.5%

    ## Why isnt it better for different prefix?
    ## I think the varying in the prefix in the different prefix is due to not enough data

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
