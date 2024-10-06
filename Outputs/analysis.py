import pandas as pd
import numpy as np

df = pd.read_csv('output.csv')

AbsoluteDifference = df['AbsoluteDifference']
PercentageDifference = df['PercentageDifference']
TimeDeviation = df['TimeDeviation']

AbsoluteDifference = np.array(AbsoluteDifference)
PercentageDifference = np.array(PercentageDifference)
TimeDeviation = np.array(TimeDeviation)

Mean_AD = np.mean(AbsoluteDifference)
Mean_PD = np.mean(PercentageDifference)
Mean_TD = np.mean(TimeDeviation)

range_AD = max(AbsoluteDifference) - min(AbsoluteDifference)
range_PD = max(PercentageDifference) - min(PercentageDifference)
range_TD = max(TimeDeviation) - min(TimeDeviation)

extreme = 0
NormilzedMeanPD = []
for i in PercentageDifference:
    if i > 10:
        extreme +=1
    else:
        NormilzedMeanPD.append(i)

Mean_PD = np.mean(NormilzedMeanPD)

        

print(Mean_AD, Mean_PD, Mean_TD)
print(extreme)

