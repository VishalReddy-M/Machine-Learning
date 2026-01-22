import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("HeartFailureDataset.csv")
print(df.isna().sum())
df['age'] = (df['age'] / 365.25).round(1)
df['risk_target'] = (
    (df['cholesterol'] <= 3) &
    (df['gluc'] <= 3) &
    (df['alco'] == 1) &
    (df['smoke'] == 1) &
    (df['age'] >= 55)
).astype(int)
#construct box plots to check for outliers
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='risk_target', y='age', data=df)
plt.title('Box Plot of Age by Risk Target')
plt.subplot(1, 2, 2)
sns.boxplot(x='risk_target', y='cholesterol', data=df)
plt.title('Box Plot of Cholesterol by Risk Target')
plt.tight_layout()
plt.show()  
#construct a heatmap to visualize correlations
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
#drop irrelevant columns
df = df.drop(columns=['id', 'height', 'weight','active','cardio'])
print(df)
#model definition
x= df.drop("risk_target",axis=1)
y=df["risk_target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)
#model building
model = LogisticRegression()
#fitting on the training data
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
con = confusion_matrix(y_test,y_pred)
print(acc)
print(con)
new_data = np.array([[65,2,140,90,3,3,1,1]])
predict = model.predict(new_data)
print(predict)
if predict[0]==1:
    print("risk_target: patient is in high risk")
else:
    print("risk_target:patient is in low risk")

