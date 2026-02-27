#K = no of clusters
# if K=2 then the data split into 2 groups
# K is not default it has to be assigned before the implementation
# cluster has own centroid
# the data is divided into K clusters such that each data point belongs to the cluster with the nearest mean(centroid)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder,StandardScaler
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
print(df.describe())
print(df.isna().sum())
label = LabelEncoder()
df["Gender"] = label.fit_transform(df["Gender"])
plt.figure(figsize=(6,6))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm") #viridis,plasma,coolwarm,blues,purples are cmaps
plt.title("corelation",fontsize = 25)
plt.xlabel("x axis features")
plt.ylabel("y axis features")
plt.show()
#feature engineering
x = df[["Annual Income (k$)","Spending Score (1-100)"]] #No Y values
scale = StandardScaler()
x_scaled = scale.fit_transform(x)
#Elbow method
#WCSS --> within cluster sum of squares
wcss = []
for k in range(1,11):
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++", #smart centroid initialization /prevents bad clustering
        random_state=42
    )
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_) # it gives wcss value and it rectifies errors inside the 
plt.plot(range(1,11),wcss,marker = "o")
plt.xlabel("the no of clusters")
plt.ylabel("WCSS")
plt.title("ELBOW METHOD")
plt.show()
model = KMeans(n_clusters=5,init="k-means++",random_state=42)
y = model.fit_predict(x_scaled)
df["cluster"] = y
plt.figure(figsize=(7,7))
plt.scatter(x_scaled[y==0,0],x_scaled[y==0,1],label = "cluster0")
plt.scatter(x_scaled[y==1,0],x_scaled[y==1,1],label = "cluster1")
plt.scatter(x_scaled[y==2,0],x_scaled[y==2,1],label = "cluster2")
plt.scatter(x_scaled[y==3,0],x_scaled[y==3,1],label = "cluster3")
plt.scatter(x_scaled[y==4,0],x_scaled[y==4,1],label = "cluster4")
plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    s = 200,
    c = "black",
    label = "centrioids"
)
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.title("customer segmentation",fontsize = 45,color = "red")
plt.legend()
plt.show()
new_data = [[100,95]]
new_data_scaled = scale.transform(new_data)
prediction = model.predict(new_data_scaled)
print(prediction)
plt.figure(figsize=(8,7))
for i in range(5):
    plt.scatter(x_scaled[y==i,0],x_scaled[y==i,1],label = f"cluster{i}")
plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    s = 200,
    c = "black",
    label = "centrioids"
)
plt.scatter(
    new_data_scaled[0,0],
    new_data_scaled[0,1],
    s=300,
    marker="*",
    label = "new_customer"
)
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.title("customer segmentation",fontsize = 45,color = "red")
plt.legend()
plt.show()