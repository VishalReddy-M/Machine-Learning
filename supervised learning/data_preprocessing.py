import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = {'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 75000, 90000, 100000000],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'buys': ['yes', 'no', 'yes', 'no', 'yes']}
df = pd.DataFrame(data)
print(df)
print(df.shape)
print(df.info())
print(df.describe())
#Conversion Using LabelEncoder
label = LabelEncoder()
df["buys"] = label.fit_transform(df["buys"])
print(df)
#Using One-Hot Encoding
df = pd.get_dummies(df,columns=["city"],drop_first=False) 
print(df)
#Feature Scaling,for mean centered algorithms 
from sklearn.preprocessing import StandardScaler
scalar= StandardScaler()
df[["age","salary"]]=scalar.fit_transform(df[["age","salary"]])
print(df)
#Normalization , distance based models ,data-point to another data-point (between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
df[["age","salary"]] = mm.fit_transform(df[["age","salary"]])
print(df)
