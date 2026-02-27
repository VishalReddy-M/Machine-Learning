import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
df = pd.read_csv("train.csv")
print(df.head())
print(df.info())
print(df.isna().sum())
plt.figure(figsize=(6,5))
sns.countplot(x="label",data=df)
plt.title("sentiment distribution")
plt.xlabel("sentiment")
plt.ylabel("count")
plt.show()
df["tweet_length"] = df["tweet"].apply(len)
print(df["tweet_length"])
plt.figure(figsize=(6,5))
sns.histplot(df["tweet_length"],bins=50,kde=True)
plt.title("tweet length distribution")
plt.xlabel("tweet length")
plt.ylabel("frequency")
plt.grid()
plt.show()

def clean_texted(text):
    text = re.sub('[^a-zA-Z]','',text)
    text = text.lower()
    text = re.sub('\s+',' ',text).strip()
    return text 
df["clean_texted"] = df["tweet"].apply(clean_texted)
print(df["clean_texted"])
x = df["clean_texted"]
y= df["label"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43,stratify=y)
vec = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    max_features=15000,
    min_df=3
)
x_train_vec = vec.fit_transform(x_train)
x_test_vec = vec.transform(x_test)
#hyper parameter tuning
params = {
    'alpha':[1.0,0.5,0.1,0.01]
}
grid = GridSearchCV(MultinomialNB(),params,scoring = "accuracy",cv=5)
grid.fit(x_train_vec,y_train)
model = grid.best_estimator_
y_pred = model.predict(x_test_vec)
print(accuracy_score(y_test,y_pred))
new_text = [
    "thankyou user for you follow"
]
new_text_clean = [clean_texted(text) for text in new_text ]
new_text_vec = vec.transform(new_text_clean)
prediction = model.predict(new_text_vec)
print(prediction)
probability = model.predict_proba(new_text_vec)
for text,label,probability in zip(new_text,prediction,probability):
    print("text",text)
    print("prediction",label)
    print("probablilty",round(max(probability)*100,2))