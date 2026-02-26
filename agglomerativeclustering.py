import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
data = load_iris()
x = data.data[:,:2]
df = pd.DataFrame(x,columns=[["sepal_length","sepal_width"]])
#print(df.head())
#print(df.isna().sum())
iris = load_iris()
df1 = pd.DataFrame(iris.data,columns=iris.feature_names)
df1["target"] = iris.target
count= df1["target"].value_counts()
#print(count)
explode = (0.05,0.05,0.05)
plt.pie(count,labels=["setosa","versicolor","virginica"],autopct="%1.1f%%",explode=explode,
        shadow=True)
plt.title("pie chart _iris",fontsize = 45)
plt.show()