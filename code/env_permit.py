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


df = df[df["Prefix"] <= 21]

df["Ground truth times"] = df["Ground truth times"] / (60 * 60 * 24)
df["Predicted times"] = df["Predicted times"] / (60 * 60 * 24)
df["MAE"] = df["MAE"] / (60 * 60 * 24)



for k in range(2, 22):
    helperDF = df.loc[df['Prefix'] == k]
        
    Ymean = helperDF["Ground truth times"].mean()
    Ymed = helperDF["Ground truth times"].median()
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



## 1. Histogram
plt.hist(df["Prefix"], bins = 20, range=(2, 21), rwidth=0.8)
plt.xlabel('Prefix')
plt.ylabel('Number of process instances')
x_ticks = range(2, 22)
plt.xticks(x_ticks)
plt.title("Histogram of number of process instances that reach prefix")
plt.savefig("Histogram prefix env permit")
#plt.show()
plt.clf()



## 2. Histogram und schauen, was besser ist.
mask = df["MAE Mean"] > df["MAE Median"]
countBigger = mask.sum()

print("Mean has been better on")
print(countBigger)
print("for number of instances:")
print(len(df))


plt.hist(df["bestCorrelation Median"], bins = 11)
plt.xlabel('Best Confidence')
plt.ylabel('Number of process instances')
plt.title("Histogram of best Confidence")
plt.savefig("Histogram best confidence env permit")
#plt.show()
plt.clf()


## 3. Improvement
improvementArray = []
for k in range(2, 22):
    helperDf = df.loc[df['Prefix'] == k]
    MAEBefore = helperDf["MAE"].mean()
    MAEAfter = helperDf["bestCorrelation Mean"].mean()

    improvement = MAEAfter / MAEBefore
    improvementArray.append(improvement * 100)

x_values = [i + 2 for i in range(len(improvementArray))]
plt.scatter(x_values, improvementArray)
plt.xlabel('Prefix length')
plt.ylabel('Improvement in %')
plt.title("Improvement env permit")
plt.savefig("Improvement env permit")
#plt.show()
plt.clf()



## 4. Using Mean as Prediction
for j in range(2, 22):
    secondHelperDf = df.loc[df['Prefix'] == j]
    Ymean = secondHelperDf["Ground truth times"].mean()
    print("Mean second")
    print(Ymean)
    for index, row in secondHelperDf.iterrows():
        df.loc[index, "MAE regarding Mean"] = abs(Ymean - row["Ground truth times"])


averageMAE = []
averageMAEMean = []
for l in range(2, 22):
    thirdHelperDf = df.loc[df['Prefix'] == l]
    averageMAE.append(thirdHelperDf["MAE"].mean())
    averageMAEMean.append(thirdHelperDf["MAE regarding Mean"].mean())
    
index_labels = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                '14', '15', '16', '17', '18', '19', '20', '21']
bar_width = 0.35
x = np.arange(len(index_labels))
plt.bar(x - bar_width / 2, averageMAE, width=bar_width, color = 'orange', label = 'initial Prediction')
plt.bar(x + bar_width / 2, averageMAEMean, width=bar_width, label='Mean as prediction')

plt.xlabel('Prefix length')
plt.ylabel('MAE')
plt.title('MAE before adjustment vs predicting always mean for given Prefix')
plt.xticks(x, index_labels)
plt.legend()
# Show the plot
# plt.show()
plt.savefig("Improvement Mean env permit dataset")
plt.clf()








