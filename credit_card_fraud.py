import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("creditcard.csv")
#print(df.head())
#print(df.shape)
#print(df.describe())
#checking null values
#print(df.isna().sum())
#print(df.dtypes)
#model variable declaration
x = df.drop(columns=['Class', 'Time'])
y = df['Class']
#splitting the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
#standard scaling 
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
#new data
new_data = np.array([
    [-1.23,  0.45, -0.87,  1.34, -0.22,  0.18, -0.56,  0.91, -0.33, -1.12,
      0.44, -0.78,  0.15, -2.01,  0.66, -0.47, -0.88,  0.12,  0.03,  0.29,
     -0.19,  0.14, -0.08,  0.51, -0.42,  0.37,  0.09, -0.02,  45.60],

    [ 0.88, -1.34,  0.56, -0.98,  1.12, -0.44,  0.63, -0.72,  1.01,  0.56,
     -0.67,  0.44, -0.21,  0.87, -0.15,  1.02,  0.76, -0.31,  0.45, -0.18,
      0.62, -0.29,  0.34, -0.66,  0.58, -0.11,  0.04,  0.27, 120.00],

    [-2.11,  1.98, -1.45,  2.76, -0.88,  0.44, -1.56,  0.91, -1.22, -2.89,
      1.56, -2.12,  0.78, -4.11,  0.33, -1.67, -2.34, -0.98,  0.76,  0.88,
     -0.67,  0.55, -0.44,  0.12, -0.87,  0.92,  0.44, -0.21,   5.99],

    [ 0.12, -0.08,  0.19, -0.05,  0.04,  0.01, -0.02,  0.03, -0.01,  0.06,
     -0.04,  0.02,  0.01, -0.03,  0.00,  0.02, -0.01,  0.00,  0.01, -0.02,
      0.03, -0.01,  0.02,  0.00,  0.01, -0.01,  0.00,  0.02, 250.75],

    [-0.98,  0.67, -0.45,  1.23, -0.56,  0.78, -0.91,  0.34, -0.67, -1.56,
      0.88, -1.02,  0.45, -2.45,  0.78, -0.89, -1.34, -0.45,  0.22,  0.56,
     -0.34,  0.29, -0.18,  0.66, -0.55,  0.41,  0.19, -0.09,  89.30]
])
new_data_scaled= scaler.transform(new_data)
predict = model.predict(new_data_scaled)
print(predict)
if predict[0]==1:
    print("Transaction: fradulant")
else:
    print("Transaction: legitimate")





