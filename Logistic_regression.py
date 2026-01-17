#Logistic regression--> numbers(input features) + discrete(target/output)
# >=0.5 it is good
# >0.9 it is very confident that our dataset is connected with model
# <0.5 it is bad
# <0.2 it is worst

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("breast-cancer.csv")
df.drop(columns="id",inplace=True)
print(df.dtypes)
label=LabelEncoder()
df["diagnosis"] = label.fit_transform(df["diagnosis"])
print(df.head())
print(df.describe())
print(df.isna().sum())
x=df.drop("diagnosis",axis=1)
y=df["diagnosis"]
#splitting 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
#standard scalar
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#model building
model = LogisticRegression()
#fitting on the training data
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
acc = accuracy_score(y_test,y_pred)
print(acc)
#new data prediction
new_data = np.array([[
    14.2,22.2,96,650,0.08,0.11,0.038,0.07,0.099,0.06,0.45,1.2,3.5,40.3,0.006,0.02,0.03,0.11,
    0.003,16,5,30.02,110,4,0.012,0.02,0.25,0.30,0.15,0.30
]])
new_data_scaled= scaler.transform(new_data)
predict = model.predict(new_data_scaled)
print(predict)
if predict[0]==1:
    print("diagnosis: Malignant(cancer)")
else:
    print("diagnosis: Benign(no cancer)")


