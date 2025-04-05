import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

#separate test

print(50//20)

# Generate the dataset
X, y = make_blobs(n_samples=300, centers=np.random.randint(25,50), n_features=2, random_state=42)

# Perform KMeans with 8 clusters
kmeans_single = KMeans(n_clusters=8, random_state=42)
labels_single = kmeans_single.fit_predict(X)

# Perform 3 layers of 2 KMeans
kmeans_layer1 = KMeans(n_clusters=2, random_state=42)
labels_layer1 = kmeans_layer1.fit_predict(X)

kmeans_layer2_1 = KMeans(n_clusters=2, random_state=42)
labels_layer2_1 = kmeans_layer2_1.fit_predict(X[labels_layer1 == 0])
# new dataset
X_new_1 = X[labels_layer1 == 0]

kmeans_layer2_2 = KMeans(n_clusters=2, random_state=42)
labels_layer2_2 = kmeans_layer2_2.fit_predict(X[labels_layer1 == 1])
X_new_2 = X[labels_layer1 == 1]

kmeans_layer3_1 = KMeans(n_clusters=2, random_state=42)
labels_layer3_1 = kmeans_layer3_1.fit_predict(X_new_1[labels_layer2_1 == 0])
X_new_1_1 = X_new_1[labels_layer2_1 == 0]

kmeans_layer3_2 = KMeans(n_clusters=2, random_state=42)
labels_layer3_2 = kmeans_layer3_2.fit_predict(X_new_1[labels_layer2_1 == 1])
X_new_1_2 = X_new_1[labels_layer2_1 == 1]

kmeans_layer3_3 = KMeans(n_clusters=2, random_state=42)
labels_layer3_3 = kmeans_layer3_3.fit_predict(X_new_2[labels_layer2_2 == 0])
X_new_2_1 = X_new_2[labels_layer2_2 == 0]

kmeans_layer3_4 = KMeans(n_clusters=2, random_state=42)
labels_layer3_4 = kmeans_layer3_4.fit_predict(X_new_2[labels_layer2_2 == 1])
X_new_2_2 = X_new_2[labels_layer2_2 == 1]


# Check if the results are the same
same_results = np.array_equal(labels_single, labels_layer2_2)

# Plot the results
plt.figure(figsize=(12, 4))

plt.subplot(141)
plt.scatter(X[:, 0], X[:, 1], c=labels_single)
plt.title("Single KMeans with 8 clusters")

plt.subplot(142)
plt.scatter(X[labels_layer1 == 0, 0], X[labels_layer1 == 0, 1], c=labels_layer2_1)
plt.scatter(X[labels_layer1 == 1, 0], X[labels_layer1 == 1, 1], c=2+labels_layer2_2)
plt.title("2 Layers of 2 KMeans")



plt.subplot(143)
plt.scatter(X_new_1_1[:, 0], X_new_1_1[:, 1], c=labels_layer3_1)
plt.scatter(X_new_1_2[:, 0], X_new_1_2[:, 1], c=2+labels_layer3_2)
plt.scatter(X_new_2_1[:, 0], X_new_2_1[:, 1], c=4+labels_layer3_3)
plt.scatter(X_new_2_2[:, 0], X_new_2_2[:, 1], c=6+labels_layer3_4)

plt.title("3 Layers of 2 KMeans")

plt.subplot(144)
plt.text(0.5, 0.5, f"Same Results: {same_results}", fontsize=14, ha='center')
plt.axis('off')

plt.tight_layout()
plt.show()