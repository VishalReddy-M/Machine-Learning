#BAGGING
'''it is also known as bootstrap aggregating
it means training many models instead of one and combining their answers to get a better and more stable result'''
''' it goes on majority of voting in classification(yes/no)
    it goes on average of voting in regression (house price predictions)'''
'''SYNTAX:
from sklearn.ensemble import RandomForestClassification
model=RandomForestClassification(parameters....,bootstrap=True)'''
#BOOSTING
'''it is also an ensemble learning technique where models are trained one after another each new model focuses on 
correcting the mistake made by the previous model'''
'''it is also known as sequential model'''
#RANDOM FOREST ALGORITHM
#parameters:
'''1.n_estimators
    no of trees(100-300--> balanced) default:100
   2.max_depth
    it says depth of trees(small(10->20) is preferred),default:None,depends on dataset size
   3.min_samples_split
    min samples needed to split a node,default:2,limit:int>=2
   4.min_samples_leaf:
    min samples of leaf node,default:1,limit:int>=1
   5.bootstrap
    whether sampling should follow the randomness and bagging ,default:True Mandatory
   6.criterion
    classification:gini,entropy
    regression:squared_error,poission
   7.random_state
    default:none,limit:int>=40'''

#IMPLEMENTATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import plot_tree
df = pd.read_csv("Titanicdataset.csv")
print(df.head())
print(df.isna().sum())
print(df.info())
df1=df[['Survived','Sex','Pclass','SibSp','Parch','Fare','Age']]
df1["Age"] = df1["Age"].fillna(df1["Age"].mean())
print(df1.isna().sum())
#preprocessing
label = LabelEncoder()
df1['Sex'] = label.fit_transform(df1['Sex'])
print(df1.head())
#feature engineering
x=df1.drop("Survived",axis=1)
y=df1["Survived"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
#model building
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    criterion="gini",
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    bootstrap=True
)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
new_data = [[1,1,0,0,73,70]]
prediction=model.predict(new_data)
if prediction[0] == 1:
    print("passenger is survived")
else:
    print("passenger is not survived")

feature_names = x.columns.tolist()
class_names = ['Not Survived', 'Survived']
plt.figure(figsize=(20,10))
plot_tree(model.estimators_[0], feature_names=feature_names, class_names=class_names, filled=True, rounded=True,fontsize=10)
plt.show()