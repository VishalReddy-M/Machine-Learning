import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
data = load_diabetes()
x=data.data
y=data.target
df =pd.DataFrame(x,columns=data.feature_names)
df["diabetes"] = y
#print(df.head())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.fit_transform(x_test)
model =Ridge(alpha=1.0) # stable alpha value
model.fit(x_train_scaled,y_train)
y_pred= model.predict(x_test)
print(model.coef_)
lr = LinearRegression()
lr.fit(x_train_scaled,y_train)
print(lr.coef_)