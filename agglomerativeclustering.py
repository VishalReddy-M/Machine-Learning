import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
data = load_iris()
x = data.data[:,:2]
df = pd.DataFrame(x,columns=[["sepal_length","sepal_width"]])
print(df.head())
print(df.isna().sum())
iris = load_iris()
df1 = pd.DataFrame(iris.data,columns=iris.feature_names)
df1["target"] = iris.target
count= df1["target"].value_counts()
#print(count)
explode = (0.09,0.05,0.05)
plt.pie(count,labels=["setosa","versicolor","virginica"],autopct="%1.1f%%",explode=explode,
        shadow=True)
plt.title("pie chart _iris",fontsize = 45)
plt.show()
scale = StandardScaler()
x_scaled = scale.fit_transform(df)
#dendrogram with linkage
z = linkage(x_scaled,method="ward")
plt.figure(figsize=(8,8))
dendrogram(z)
plt.title("agglomerative clustering dendrogram")
plt.xlabel("the samples index")
plt.ylabel("distance")
plt.tight_layout()
plt.show()
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(x_scaled)
plt.figure(figsize=(6,6))
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=labels,cmap="viridis")
plt.title("agglomerative clustering")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()
print(silhouette_score(x_scaled,labels))
print(davies_bouldin_score(x_scaled,labels))
print(calinski_harabasz_score(x_scaled,labels))
new_data = [[5.7,3.4]]
new_data_scaled = scale.transform(new_data)
plt.figure(figsize=(6,6))
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=labels,cmap="viridis")
plt.scatter(new_data_scaled[:,0],new_data_scaled[:,1],s=200,marker="X",color="red")
plt.title("agglomerative clustering")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()
df["cluster"] = labels
