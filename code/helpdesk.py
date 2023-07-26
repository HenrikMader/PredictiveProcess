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

df = pd.read_csv("./output_files/results/helpdesk/suffix_and_remaining_time_helpdesk.csv")
df = df.rename(columns={"Prefix length": "Prefix"})

# This range is set to maximal Prefix length + 1
maximalRange = 8


## Note: If you want to generate the filtered dataframe for Prefix length 2, then uncomment the following four lines out. This will change
## the predictions of result 1, 2 and 3.
#caseID = df.loc[df['Prefix'] == 4]["CaseID"]
#df = df[df['CaseID'].isin(caseID)]
#df = df[df["Prefix"] <= 4]
#maximalRange = 5




df["Ground truth times"] = df["Ground truth times"] / (60 * 60 * 24)
df["Predicted times"] = df["Predicted times"] / (60 * 60 * 24)
df["MAE"] = df["MAE"] / (60 * 60 * 24)

df = df.drop(['Groud truth', 'Predicted', 'Levenshtein', 'Damerau', 'Jaccard', 'RMSE'], axis=1)

for k in range(2, maximalRange):
    helperDF = df.loc[df['Prefix'] == k]
        
    Ymean = helperDF["Ground truth times"].mean()
    Ymed = helperDF["Ground truth times"].median()

    print("Mean first")
    print(Ymean)
    for index, row in helperDF.iterrows():
        bestCorrelationMean = 0
        errorMean = 100000000000
        bestCorrelationMedian = 0
        errorMedian = 100000000000
        for i in range(0, 11, 1):
            c = i / 10
            df.loc[index, str(c) + " mean"] = (c * row["Predicted times"]) + (Ymean * (1 - c))
            if (abs((((c * row["Predicted times"]) + (Ymean * (1 - c)) - row["Ground truth times"]))) < errorMean):
                errorMean = (abs((c * row["Predicted times"]) + (Ymean * (1 - c)) - row["Ground truth times"]))
                bestCorrelationMean = c

            df.loc[index, str(c) + " median"] = (c * row["Predicted times"]) + (Ymed * (1 - c))
            if (abs((((c * row["Predicted times"]) + (Ymed * (1 - c)) - row["Ground truth times"]))) < errorMedian):
                errorMedian = (abs((c * row["Predicted times"]) + (Ymed * (1 - c)) - row["Ground truth times"]))
                bestCorrelationMedian = c

        df.loc[index, "bestCorrelation Mean"] = bestCorrelationMean
        df.loc[index, "MAE Mean"] = abs((bestCorrelationMean * row["Predicted times"] + (1 - bestCorrelationMean) * Ymean) - row["Ground truth times"])
        df.loc[index, "bestCorrelation Median"] = bestCorrelationMedian
        df.loc[index, "MAE Median"] = abs((bestCorrelationMedian * row["Predicted times"] + (1 - bestCorrelationMedian) * Ymed) - row["Ground truth times"])
           




## 1. Histogram for Prefix Helpdesk
plt.hist(df["Prefix"], bins = 6, range=(2, maximalRange), rwidth=0.8)
plt.xlabel('Prefix')
plt.ylabel('Number of process instances')
x_ticks = range(2, maximalRange)
plt.xticks(x_ticks)
plt.title("Histogram of number of process instances that reach prefix")
plt.savefig("Histogram prefix helpdesk")
#plt.show()
plt.clf()



## 2. Histogram best confidence
mask = df["MAE Mean"] > df["MAE Median"]
countBigger = mask.sum()

print("Mean has been better on")
print(countBigger)
print("for number of instances:")
print(len(df))



plt.hist(df["bestCorrelation Mean"], bins = 11)
plt.xlabel('Best Confidence')
plt.ylabel('Number of process instances')
plt.title("Histogram of best Confidence")
plt.savefig("Histogram best confidence helpdesk")
#plt.show()
plt.clf()


## 3. Improvement
improvementArray = []
for k in range(2, maximalRange):
    helperDf = df.loc[df['Prefix'] == k]
    MAEBefore = helperDf["MAE"].mean()
    MAEAfter = helperDf["bestCorrelation Mean"].mean()

    improvement = MAEAfter / MAEBefore
    improvementArray.append(improvement * 100)

x_values = [i + 2 for i in range(len(improvementArray))]
plt.scatter(x_values, improvementArray)
plt.xlabel('Prefix length')
plt.ylabel('Improvement in %')
plt.title("Improvement helpdesk")
plt.savefig("Improvement helpdesk")
plt.clf()


## 4. Using Mean as Prediction
for j in range(2, maximalRange):
    secondHelperDf = df.loc[df['Prefix'] == j]
    Ymean = secondHelperDf["Ground truth times"].mean()
    print("Mean second")
    print(Ymean)
    for index, row in secondHelperDf.iterrows():
        df.loc[index, "MAE regarding Mean"] = abs(Ymean - row["Ground truth times"])

averageMAE = []
averageMAEMean = []
for l in range(2, maximalRange):
    thirdHelperDf = df.loc[df['Prefix'] == l]
    averageMAE.append(thirdHelperDf["MAE"].mean())
    averageMAEMean.append(thirdHelperDf["MAE regarding Mean"].mean())
    
index_labels = ['Prefix {}'.format(l) for l in range(2, maximalRange)]

bar_width = 0.35
x = np.arange(len(index_labels))

plt.bar(x - bar_width / 2, averageMAE, width=bar_width, color='orange', label='Initial Prediction')
plt.bar(x + bar_width / 2, averageMAEMean, width=bar_width, label='Mean as Prediction')

plt.xlabel('Prefix length')
plt.ylabel('MAE')
plt.title('MAE before adjustment vs predicting always mean for given Prefix')
plt.xticks(x, index_labels)
plt.legend()
plt.tight_layout()
plt.savefig("Mean for filtered dataset helpdesk dataset")
#plt.show()



