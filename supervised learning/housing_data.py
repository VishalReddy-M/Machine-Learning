import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")

house = fetch_california_housing()
df = pd.DataFrame(house.data,columns=house.feature_names)
y= house.target
df["Price"]=y
print(df)
print(df.isna().sum())
print(df.describe())

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.figure(figsize=(16, 10))

# 1. Distribution of Price (Histogram with KDE)
plt.subplot(2, 2, 1)
plt.hist(df['Price'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Housing Prices')

# 2. Correlation Heatmap
plt.subplot(2, 2, 2)
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()

# 3. Median Income vs Price (Scatter Plot)
plt.subplot(2, 2, 3)
plt.scatter(df['MedInc'], df['Price'], alpha=0.5, s=30, color='darkblue')
plt.xlabel('Median Income')
plt.ylabel('Price')
plt.title('Median Income vs Housing Price')
plt.grid(True, alpha=0.3)

# 4. Geographic Distribution (Latitude/Longitude colored by Price)
plt.subplot(2, 2, 4)
scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['Price'], cmap='viridis', alpha=0.6, s=30)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of Housing Prices')
cbar = plt.colorbar(scatter)
cbar.set_label('Price')

plt.tight_layout()
plt.show()

x = df.drop("Price", axis=1)
y = df["Price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)

model = DecisionTreeRegressor()
model.fit(x_train,y_train)
pred = model.predict(x_test)

print(mean_squared_error(y_test,pred))
print(r2_score(y_test,pred))

new_data = [
    [6.2, 30, 7.1, 1.0, 890, 2.9, 34.05, -138.25],
    [5.2, 28, 6.1, 1.0, 850, 4.9, 35.05, -118.25],
    [3.8, 15, 5.4, 1.1, 1200, 3.2, 37.77, -122.42]
]
prediction = model.predict(new_data)
print("the predicted price of House in California",prediction)


