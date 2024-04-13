import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset from CSV
df = pd.read_csv("D:\docs\Study-Materials-BVM-22CP308\sem 6\python ML\machine-learning\lab 5\IRIS.csv")

# Extract features
X = df.iloc[:, :-1].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Hierarchical Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_cluster.fit(X_scaled)

# Visualize the clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agg_cluster.labels_, cmap='viridis', s=50, alpha=0.7)
plt.title('Hierarchical Agglomerative Clustering on Iris Dataset')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.show()
