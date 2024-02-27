import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score

data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_test_pred = knn_model.predict(X_test)
y_train_pred = knn_model.predict(X_train)
#test
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred,average='macro')
recall_test = recall_score(y_test, y_test_pred,average='macro')
#train
precision_train = precision_score(y_train, y_train_pred,average='macro')
accuracy_train = accuracy_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred,average='macro')

print("Accuracy of the KNN model on the test data:", accuracy_test)
print("Precision of the KNN model on the test data:", precision_test)
print("Recall of the KNN model on the test data:", recall_test)

print("Accuracy of the KNN model on the train data:", accuracy_train)
print("Precision of the KNN model on the train data:", precision_train)
print("Recall of the KNN model on the train data:", recall_train)


plt.figure(figsize=(10, 6))
targets = np.unique(y)
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indices_to_keep = y == target
    plt.scatter(X_pca[indices_to_keep, 0], X_pca[indices_to_keep, 1], c=color, s=50)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(targets, loc='best')
plt.title('PCA of Wine Dataset')
plt.show()
