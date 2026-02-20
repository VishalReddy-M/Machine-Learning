# GAUSSIAN NAIVE BAYES ALGORITHM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv("weather.csv")
print(df.shape)
print(df.describe())
# create per-column LabelEncoders and fit
le_outlook = LabelEncoder()
df["Outlook"] = le_outlook.fit_transform(df["Outlook"])
le_temp = LabelEncoder()
df["Temp"] = le_temp.fit_transform(df["Temp"])
le_humidity = LabelEncoder()
df["Humidity"] = le_humidity.fit_transform(df["Humidity"])
le_windy = LabelEncoder()
df["Windy"] = le_windy.fit_transform(df["Windy"])
le_play = LabelEncoder()
df["Play"] = le_play.fit_transform(df["Play"])
print(df.head())
x= df.drop("Play",axis=1)
y=df["Play"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#model building
model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# create new input and normalize to lowercase to match training
new_data = pd.DataFrame({
    "Outlook":["Rainy"],
    "Temp":["Cool"],
    "Humidity":["High"],
    "Windy":["f"]
})

feature = {
    "Outlook":le_outlook,
    "Temp":le_temp,
    "Humidity":le_humidity,
    "Windy":le_windy
}
for cols in new_data.columns:
    new_data[cols] = feature[cols].transform(new_data[cols])

prediction = model.predict(new_data)
print(prediction)
result = le_play.inverse_transform(prediction)
print(result)  