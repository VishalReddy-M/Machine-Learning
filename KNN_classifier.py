import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

data = load_wine()
x=data.data
y=data.target
df = pd.DataFrame(x,columns=data.feature_names)
df["target"] = y
#print(df.head())
#print(df.describe())
#feature engineering
x = df.drop("target",axis=1)
y = df["target"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#standard scaling
scale = StandardScaler()
x_train_scaled=scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)
#model
model = KNeighborsClassifier(n_neighbors=4,weights="uniform",algorithm="auto",metric="minkowski",p=2)
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
new_data = [[
    13.5,2.02,2.56,18.6,110,2.9,3.0,0.4,2.20,5.56,1.06,3.23,1056.0
]]
new_data_scaled = scale.transform(new_data)
prediction = model.predict(new_data_scaled)
print(prediction)
#PCA 
#where high dimensional features are converted into 2D features as x,y
pca = PCA(n_components=2)
x_scaled = scale.fit_transform(x)
x_pca = pca.fit_transform(x_scaled)
new_pca = pca.transform(new_data_scaled)
plt.figure()
for class_value in np.unique(y):
    plt.scatter(
        x_pca[y==class_value,0],
        x_pca[y==class_value,1],
        label = class_value
    )
# plot the new sample on top
plt.scatter(
    new_pca[:,0],
    new_pca[:,1],
    label = "new_data_scaled"
)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA visualization")
plt.legend()
plt.show()