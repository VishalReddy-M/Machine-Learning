import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
x= iris.data
print(x)
y=iris.target
df=pd.DataFrame(x,columns=iris.feature_names)
print(df)
df['Target']=y
print(y)
