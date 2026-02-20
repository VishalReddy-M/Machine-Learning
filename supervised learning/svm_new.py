import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Load iris dataset
iris = load_iris()

#CHANGE 1: Use all 4 features
X = iris.data
y = iris.target

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#CHANGE 2: Apply PCA for dimensionality reduction (4D → 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train SVM on original 4D scaled data
model = SVC(kernel="linear", C=1)
model.fit(X_scaled, y)

# Create meshgrid in PCA space
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

#CHANGE 3: Convert PCA grid back to 4D before prediction
grid = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

print("Hyperplane equation:")
print(model.coef_, "x +", model.intercept_)

# Plot decision regions
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, alpha=0.5)

# Plot data points
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
            label="class 0", edgecolors="k")
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
            label="class 1", edgecolors="k")
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1],
            label="class 2", edgecolors="k")

#CHANGE 4: New input must have 4 features
new_data = [[5.8, 3.1, 4.0, 1.2]]
new_data_scaled = scaler.transform(new_data)
new_data_pca = pca.transform(new_data_scaled)
prediction = model.predict(new_data_scaled)

# Plot new data point
plt.scatter(
    new_data_pca[0, 0],
    new_data_pca[0, 1],
    c="blue",
    s=200,
    marker="X",
    label=f"(predicted class: {prediction[0]})"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM with 4 Features (Visualized using PCA)")
plt.legend()
plt.show()

