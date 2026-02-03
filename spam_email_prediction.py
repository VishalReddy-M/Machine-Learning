import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

df = pd.read_csv("emails.csv")
#print(df.shape)
#print(df.isna().sum())
df["cleaned_text"] = df["text"].str.lower()
vectorizer  = CountVectorizer(binary=True)
x= vectorizer.fit_transform(df["cleaned_text"])
y = df["spam"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = BernoulliNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
new_text = [
    "congratulations! you won a lottery",
    "attend meeting in time",
    "free coupons are available,claim it fast",
    "please share your OTP"
]
new_text = [m.lower() for m in new_text]
new_text_vec = vectorizer.transform(new_text)
prediction = model.predict(new_text_vec)
print(prediction)
for text,pre in zip(new_text,prediction):
    print(f"the message :{text} \n predicted label : {pre}")