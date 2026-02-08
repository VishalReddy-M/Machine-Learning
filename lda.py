import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
data = load_breast_cancer()
x=data.data
y=data.target
print("target class",np.unique(y))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)
lda = LinearDiscriminantAnalysis(n_components=1)
x_train_lda = lda.fit_transform(x_train_scaled,y_train)
x_test_lda = lda.transform(x_test_scaled)
print(x_train_lda.shape)
print(x_test_lda.shape)
model = LogisticRegression()
model.fit(x_train_lda,y_train)
y_pred = model.predict(x_test_lda)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
