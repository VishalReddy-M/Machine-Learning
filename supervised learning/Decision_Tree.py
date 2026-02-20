#decision tree
'''
Question ----> Answer ---> Another Question ---> Final Decision
'''
'''makes decisons by asking and answering the questions by splitiing data step by step
untill the final decision is reached.
It is simply a flow chart or questionnaire,rule based model.'''
#STRUCTURE:
'''1.Root Node
    2.Internal Node
    3.Branch
    4.Leaf Node'''

#GINI INDEX 
#CRITERION
from sklearn.tree import DecisionTreeClassifier,plot_tree
model = DecisionTreeClassifier(criterion="gini") #Default
model = DecisionTreeClassifier(criterion="entropy") 
#gini ----> measures how mixed the classes are
#entropy ----> checks the tree is in ordered or not

#MAX_DEPTH
model = DecisionTreeClassifier(max_depth=2)
#limit how deep the tree can grow
#small data ---> small depth
#large data ---> large depth

#MIN_SAMPLES_SPLIT
model=DecisionTreeClassifier(min_samples_split=2)
#min no of samples required to split a node
#small data -->1 or 2 min_samples_split
#large data -->20 
#very large data --> 100-500

#MIN_SAMPLES_LEAF
model = DecisionTreeClassifier(min_samples_leaf=2)
#small data -->1 or 2 min_samples_leaf
#large data -->20 
#very large data --> 50-200

#IMPLEMENTATION
from sklearn.datasets import load_iris
data = load_iris()
x= data.data
y= data.target
import pandas as pd
df=pd.DataFrame(data.data,columns=data.feature_names)
df["target"] = y
print(df)
print(df.describe())
#Feature Engineering
x=data.data
y=data.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)
model = DecisionTreeClassifier(criterion="gini",max_depth=2,min_samples_split=2,min_samples_leaf=2,random_state=42)
#fitiing on training data
model.fit(x_train,y_train)
#predicted values
y_pred = model.predict(x_test)
print(y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred)) 
print(accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt
feature_names = data.feature_names
class_names = data.target_names
plt.figure(figsize=(7,7))
plot_tree(model,feature_names=data.feature_names,class_names=data.target_names,filled=True,rounded=True)
plt.show()