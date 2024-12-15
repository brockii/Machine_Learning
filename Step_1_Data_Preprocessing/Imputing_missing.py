import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

dataset = pd.read_csv("Data.csv")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset.iloc[:,1:-1] = imputer.fit_transform(dataset.iloc[:,1:-1])
print(list(dataset.columns))
# X = dataset.values[:, :-1]
# y = dataset.values[:,-1]
# print(X)
print(dataset)

df = pd.DataFrame(data=dataset.values, columns=list(dataset.columns))
#converting to list was necessary as it was returning some sort of object
df.to_csv("Imputed_data.csv", index=False)
df.to_excel("Imputed_data.xlsx", index=False)
# dataset.to_csv()