import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("bankloan.csv")
print(df.columns)
#print(df.isna().sum())
#print(df.info())
df = df.drop(["ID","ZIP.Code"],axis=1)
print(df.head())
df["Experience"] = df["Experience"].abs()
#  Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#feature enngineering
x = df.drop("Personal.Loan",axis=1)
y=df["Personal.Loan"]
#splitting the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#model implementation
model = RandomForestRegressor(n_estimators=100,
    max_depth=4,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    bootstrap=True
)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("mean_squared error be",mse)
print("r2 score be",r2)

new_data = [[35, 10, 185, 3, 6, 2, 120, 0, 0, 1, 1]]
prediction = model.predict(new_data)
if prediction[0]==1:
    print("personal loan is granted")
else:
    print("personal loan is not granted")