import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
X[:, 1:] = imputer.fit_transform(X[:, 1:])
print(X)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
print(Y)

Y = le.fit_transform(Y)
print(Y)

from sklearn.preprocessing import OneHotEncoder,
from sklearn.compose import ColumnTransformer
OHC = ColumnTransformer(transformers=['encoder', ])