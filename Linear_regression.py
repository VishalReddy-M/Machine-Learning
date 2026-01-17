'''Linear Regression:
---> It is a supervised learning algorithm.
---> used to predict the continous values
---> by drawing a straight line through data points
example: House_price prediction
Mathematical Formula:
            y=mx+c
Machine Learning:
            y=wX+b
        where;
            y=predicted_output
            X=input feature
            w=weight(importance)
            b=bias(starting point)
-->Linear Regression = Best Line+ prediction
-->The distance between actual value and predicted values should be minimum which is generally known as error
-->We adjust weights and bias to reduce the error step by step to reach minimum error'''
#Implementation of Linear Regression.
import pandas as pd
import numpy as np
df = pd.read_csv("StudentPerformance.csv")
print(df.head())
dd=pd.set_option('display.max_columns', None)
#Data Preprocessing
print(df.isna().sum())
print(df.describe())# statistical summaries
#Conversion of categorical_data to Numerical_data using Label encoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df["Extracurricular Activities"] = label.fit_transform(df["Extracurricular Activities"])
#EDA
import matplotlib.pyplot as plt
plt.scatter(df["Previous Scores"],df["Performance Index"])
plt.xlabel("previous score")
plt.ylabel("performance index")
plt.title("previous vs performance index")
plt.show()
#Feature engineering
x = df.drop("Performance Index", axis=1)
y = df["Performance Index"]
#splitting of data into training and testing 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)
#Model Training
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")
model = LinearRegression()
#Fitting on training data
model.fit(x_train,y_train)
pred = model.predict(x_test)
# Metrics
from sklearn.metrics import mean_squared_error,r2_score
print(mean_squared_error(y_test,pred))
print(r2_score(y_test,pred))
#Deployment(Prediction on new data points)
new_student = np.array([[6,94,1,6,7]])
prediction = model.predict(new_student)
print("the predicted performance index of student",prediction)


