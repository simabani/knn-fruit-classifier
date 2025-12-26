# src/visualize_knn.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv('data/fruits.csv')
X = data[['shape', 'color_score']].values
y = data['label'].values

# Encode labels as numbers for plotting
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Train K-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y_encoded)

# Plot setup
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict class for each point in the mesh
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Define color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot training points
for idx, class_name in enumerate(class_names):
    plt.scatter(X[y_encoded == idx, 0], X[y_encoded == idx, 1],
                c=[cmap_bold(idx)], label=class_name, edgecolor='k', s=60)

plt.xlabel('Shape')
plt.ylabel('Color Score')
plt.title("K-NN Decision Boundaries (k=3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

