import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
digit = load_digits()
x= digit.data
y= digit.target
images = digit.images
print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test,img_train,img_test = train_test_split(x,y,images,test_size=0.2,random_state=43)
scale = StandardScaler()
x_train_scaled=scale.fit_transform(x_train)
x_test_scaled=scale.transform(x_test)
pca = PCA(n_components=30)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)
print(x_train.shape)
print(x_train_pca.shape)
model = LogisticRegression()
model.fit(x_train_pca,y_train)
y_pred = model.predict(x_test_pca)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
index =50
test_image = img_test[index]
test_image_flatten =x_test[index].reshape(1,-1)
test_image_scaled = scale.transform(test_image_flatten)
test_image_pca = pca.transform(test_image_scaled)
prediction = model.predict(test_image_pca)[0]
plt.imshow(test_image,cmap="grey")
plt.title(f"predicted digit: {prediction}")
plt.show()