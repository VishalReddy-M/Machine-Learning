import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
data = fetch_california_housing()
x=data.data
y=data.target
df = pd.DataFrame(x,columns=data.feature_names)
df["price"] = y
#print(df.head())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.fit_transform(x_test)
model = Lasso(alpha=0.1)
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test)
coefficients = pd.Series(model.coef_,index=data.feature_names)
print(sum(coefficients==0)) #feature ignored by model
new_data = [[5.0,25,6,2,1000,3,34,-112]]
new_data_scaled = scale.transform(new_data)
print(model.predict(new_data_scaled))