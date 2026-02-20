''' Metrics tells us how good or bad the model's predictions are compared to the actual values
classification:
1.Accuracy score
2.classification_report
3.Confusion_matrix
4.F1_score
5.Recall
6.Precision
7.ConfusionMatrixDisplay
'''
'''Regression:
1.Mean squared error
2. Root Mean squared error
3.Mean Absolute error
4.R2_score
'''

#MEAN ABSOLUTE ERROR:
from sklearn.metrics import mean_absolute_error
y_actual = [100,50,150,200]
y_pred = [100,250,290,200]
mae = mean_absolute_error(y_actual,y_pred)
print(mae)

#MEAN SQUARED ERROR
from sklearn.metrics import mean_squared_error
y_actual = [100,150,200]
y_pred = [100,250,290]
mse = mean_squared_error(y_actual,y_pred)
print(mse)

#ROOT MEAN SQUARED ERROR
from sklearn.metrics import root_mean_squared_error
y_actual = [100,150,200]
y_pred = [100,250,290]
rmse = root_mean_squared_error(y_actual,y_pred)
print(rmse)

#R2_SCORE
from sklearn.metrics import r2_score
y_actual = [100,150,200]
y_pred = [100,150,210]
r2 = r2_score(y_actual,y_pred)
print(r2)
