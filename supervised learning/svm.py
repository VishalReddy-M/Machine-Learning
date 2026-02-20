import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
iris = load_iris()
x = iris.data[:,:2]
y = iris.target
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
model = SVC(kernel="linear",C=1)
model.fit(x_scale,y)
x_min, x_max = x_scale[:,0].min() - 1, x_scale[:,0].max() + 1
y_min, y_max = x_scale[:,1].min() - 1, x_scale[:,1].max() + 1
#mesh grid is used to create every data point in the graph
#using mesh grid we can connect with data point representation
xx, yy = np.meshgrid(np.linspace(x_min,x_max,300), np.linspace(y_min,y_max,300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(8,8))
plt.contourf(xx,yy,Z,alpha=0.5)
plt.scatter(x_scale[y==0,0],x_scale[y==0,1],label = "class 0",edgecolors="k")
plt.scatter(x_scale[y==1,0],x_scale[y==1,1],label = "class 1",edgecolors="k")
plt.scatter(x_scale[y==2,0],x_scale[y==2,1],label = "class 2",edgecolors="k")
w = model.coef_[0]
b = model.intercept_[0]
x_line = np.linspace(x_min,x_max,100)
y_line = -(w[0]*x_line+b)/w[1]
plt.plot(x_line,y_line,"k--",label="hyperplane")
plt.legend()


new_data=[[5.8,3.1]]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
plt.figure(figsize=(6,6))
plt.contourf(xx,yy,Z,alpha = 0.5)
plt.scatter(x_scale[y==0,0],x_scale[y==0,1],label = "class 0",edgecolors="k")
plt.scatter(x_scale[y==1,0],x_scale[y==1,1],label = "class 1",edgecolors="k")
plt.scatter(x_scale[y==2,0],x_scale[y==2,1],label = "class 2",edgecolors="k")
w = model.coef_[0]
b = model.intercept_[0]
x_line = np.linspace(x_min,x_max,100)
y_line = -(w[0]*x_line+b)/w[1]
plt.plot(x_line,y_line,"k--",label="hyperplane")
plt.scatter(
    new_data_scaled[0,0],
    new_data_scaled[0,1],c="blue",s=200,marker = "X",label = f'(predicted class :{prediction[0]})'
)
plt.xlabel("sepallength")
plt.ylabel("sepalwidth")
plt.title("svm alogrithm")
plt.legend()
plt.show()