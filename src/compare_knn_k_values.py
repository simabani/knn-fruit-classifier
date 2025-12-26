# src/compare_knn_k_values.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('data/fruits.csv')
X = data[['shape', 'color_score']].values
y = data['label'].values

# Encode labels for model training and color mapping
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Fixed RGB color map for each fruit
cmap_light = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])  # background
point_colors = {'apple': 'red', 'banana': 'green', 'orange': 'blue'}

# Define mesh grid for plotting decision boundaries
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot setup
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
k_values = [1, 3, 5]

for i, k in enumerate(k_values):
    ax = axes[i]

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y_encoded)

    # Predict over mesh grid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot training data points
    for idx, class_name in enumerate(class_names):
        ax.scatter(
            X[y_encoded == idx, 0],
            X[y_encoded == idx, 1],
            color=point_colors[class_name],
            label=class_name if k == 1 else "",  # Only show legend once
            edgecolor='k',
            s=50
        )

    ax.set_title(f"K-NN Decision Boundary (k={k})")
    ax.set_xlabel("Shape")
    ax.set_ylabel("Color Score")
    ax.grid(True)

# Show legend only on the first subplot
axes[0].legend(title="Fruit")

plt.tight_layout()
plt.show()

