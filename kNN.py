import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, fbeta_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


data_folder = 'data/'

image_data = []
image_size = (64,64)

for label in os.listdir(data_folder):
    subfolder_path = os.path.join(data_folder, label)
    
    if os.path.isdir(subfolder_path):
        for image_file in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_file)

            image = Image.open(image_path).convert('RGB').resize(image_size)
            image_array = np.array(image).flatten()
            image_data.append((image_array, label))  

df = pd.DataFrame(image_data, columns=['Image', 'Label'])
label_map = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor':3}
df['Label_Int'] = df['Label'].map(label_map)
X = np.stack(df['Image'].values)
y = df['Label_Int'].values

X = X.reshape(X.shape[0], -1)


scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

print(" ")
print("General kNN")


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=6740)

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1, 30, 2))}

scoring = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'fbeta_score': make_scorer(fbeta_score, beta=2, average='weighted')
}

grid_search = GridSearchCV(
    knn,
    param_grid,
    scoring=scoring,
    refit='balanced_accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best K using Grid Search:", grid_search.best_params_)
print(f"Balanced Accuracy using Grid Search for K = {grid_search.best_params_}: {grid_search.best_score_}")

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
test_fbeta_score = fbeta_score(y_test, y_pred, beta=2, average='weighted')

print(f"Test Balanced Accuracy: {test_balanced_accuracy:.2f}")
print(f"Test F2 Score: {test_fbeta_score:.2f}")


plt.figure()
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='o', label='True Labels', alpha=0.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', label='Predicted Labels', alpha=0.7, edgecolor='red')
plt.title("kNN Predictions vs True Labels_original data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("kNN Predictions vs True Labels_original data.png")
plt.colorbar(scatter, label='True Labels')

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion matrix_original data.png")

#----------------------- with PCA -----------------------
print(" ")
print("kNN with dimensionality reduction using PCA")

pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X_normalized)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_pca,y, test_size=0.2, random_state=6740)

knn_pca = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1, 30, 2))}


grid_search_pca = GridSearchCV(
    knn_pca,
    param_grid,
    scoring=scoring,
    refit='balanced_accuracy',
    cv=10,
    n_jobs=-1
)

grid_search_pca.fit(X_train2, y_train2)

print("Best K using Grid Search and PCA:", grid_search_pca.best_params_)
print(f"Balanced Accuracy using Grid Search and PCA for K = {grid_search_pca.best_params_}: {grid_search_pca.best_score_}")

best_knn_pca = grid_search_pca.best_estimator_
y_pred_pca = best_knn_pca.predict(X_test2)

test_balanced_accuracy_pca = balanced_accuracy_score(y_test2, y_pred_pca)
test_fbeta_score_pca = fbeta_score(y_test2, y_pred_pca, beta=2, average='weighted')

print(f"Test Balanced Accuracy: {test_balanced_accuracy_pca:.2f}")
print(f"Test F2 Score: {test_fbeta_score_pca:.2f}")

plt.figure()
scatter = plt.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test2, cmap='viridis', marker='o', label='True Labels', alpha=0.5)
plt.scatter(X_test2[:, 0], X_test2[:, 1], c=y_pred_pca, marker='x', label='Predicted Labels', alpha=0.7, edgecolor='red')
plt.title("kNN Predictions vs True Labels_PCA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.colorbar(scatter, label='True Labels')
plt.savefig("kNN Predictions vs True Labels_PCA.png")

cm = confusion_matrix(y_test2, y_pred_pca)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix PCA")
plt.savefig("confusion matrix_PCA.png")


#----------------------- with tSNE -----------------------
print(" ")
print("kNN with dimensionality reduction using t-SNE")

tsne = TSNE(n_components=2, random_state=6740)
X_tsne = tsne.fit_transform(X_normalized)

X_train3, X_test3, y_train3, y_test3 = train_test_split(X_tsne,y, test_size=0.2, random_state=6740)

knn_tsne = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1, 30, 2))}


grid_search_tsne = GridSearchCV(
    knn_tsne,
    param_grid,
    scoring=scoring,
    refit='balanced_accuracy',
    cv=10,
    n_jobs=-1
)

grid_search_tsne.fit(X_train3, y_train3)

print("Best K using Grid Search and t-SNE:", grid_search_tsne.best_params_)
print(f"Balanced Accuracy using Grid Search and t-SNE for K = {grid_search_tsne.best_params_}: {grid_search_tsne.best_score_}")

best_knn_pca = grid_search_tsne.best_estimator_
y_pred_tsne = best_knn_pca.predict(X_test3)

test_balanced_accuracy_tsne = balanced_accuracy_score(y_test3, y_pred_tsne)
test_fbeta_score_tsne = fbeta_score(y_test3, y_pred_tsne, beta=2, average='weighted')

print(f"Test Balanced Accuracy: {test_balanced_accuracy_tsne:.2f}")
print(f"Test F2 Score: {test_fbeta_score_tsne:.2f}")

plt.figure()
scatter = plt.scatter(X_test3[:, 0], X_test3[:, 1], c=y_test3, cmap='viridis', marker='o', label='True Labels', alpha=0.5)
plt.scatter(X_test3[:, 0], X_test3[:, 1], c=y_pred_tsne, marker='x', label='Predicted Labels', alpha=0.7, edgecolor='red')
plt.title("kNN Predictions vs True Labels_t-SNE")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.colorbar(scatter, label='True Labels')
plt.savefig("kNN Predictions vs True Labels_t-SNE.png")

cm = confusion_matrix(y_test3, y_pred_tsne)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix t-SNE")
plt.savefig("confusion matrix_t-SNE.png")


#----------------------- with epsilon-ISOMAP -----------------------
print(" ")
print("kNN with dimensionality reduction using epsilon-ISOMAP")

epsilon_range = np.linspace(0.1, 10, 100)
n_components = 2
results = {}
pairwise_distances = squareform(pdist(X_normalized, metric='euclidean'))

for epsilon in epsilon_range:
    epsilon_graph = np.where(pairwise_distances <= epsilon, pairwise_distances, np.inf)
    
    n_connected_components, labels = connected_components(epsilon_graph, directed=False)
    
    if n_connected_components > 1:
        results[epsilon] = {'connected': False, 'stress': None}
        continue
    
    geodesic_distances = shortest_path(epsilon_graph, method='D', directed=False)
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=6740)
    embedding = mds.fit_transform(geodesic_distances)
    
    stress = mds.stress_
    results[epsilon] = {'connected': True, 'stress': stress}

# best_epsilon = min(
#     (eps for eps, res in results.items() if res['connected']),
#     key=lambda eps: results[eps]['stress']
# )

print("model failed to find epsilon resulting in graph connectivity")
# print(f"Best epsilon: {best_epsilon}")

# pairwise_distances = squareform(pdist(X_normalized, metric='euclidean'))
# epsilon_graph = np.where(pairwise_distances <= best_epsilon, pairwise_distances, np.inf)
# mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=6740)
# X_isomap = mds.fit_transform(geodesic_distances)

# X_train4, X_test4, y_train4, y_test4 = train_test_split(X_isomap,y, test_size=0.2, random_state=6740)

# knn_isomap = KNeighborsClassifier()
# param_grid = {'n_neighbors': list(range(1, 30, 2))}


# grid_search_isomap = GridSearchCV(
#     knn_pca,
#     param_grid,
#     scoring=scoring,
#     refit='balanced_accuracy',
#     cv=10,
#     n_jobs=-1
# )

# grid_search_isomap.fit(X_train4, y_train4)

# print("Best K using Grid Search and ISOMAP:", grid_search_isomap.best_params_)
# print(f"Balanced Accuracy using Grid Search and ISOMAP for K = {grid_search_isomap.best_params_}: {grid_search_isomap.best_score_}")

# best_knn_isomap = grid_search_isomap.best_estimator_
# y_pred_isomap = best_knn_isomap.predict(X_test4)

# test_balanced_accuracy_isomap = balanced_accuracy_score(y_test4, y_pred_isomap)
# test_fbeta_score_isomap = fbeta_score(y_test4, y_pred_isomap, beta=2, average='weighted')

# print(f"Test Balanced Accuracy: {test_balanced_accuracy_isomap:.2f}")
# print(f"Test F2 Score: {test_fbeta_score_isomap:.2f}")

# plt.figure()
# scatter = plt.scatter(X_test4[:, 0], X_test4[:, 1], c=y_test4, cmap='Blues', marker='o', label='True Labels', alpha=0.5)
# plt.scatter(X_test4[:, 0], X_test4[:, 1], c=y_pred_isomap, marker='x', label='Predicted Labels', alpha=0.7, edgecolor='red')
# plt.title("kNN Predictions vs True Labels_ISOMAP")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend()
# plt.colorbar(scatter, label='True Labels')
# plt.savefig("kNN Predictions vs True Labels_ISOMAP.png")

# cm = confusion_matrix(y_test4, y_pred_isomap)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix ISOMAP")
# plt.savefig("confusion matrix_isomap.png")


#----------------------- with kNN-ISOMAP -----------------------
print(" ")
print("kNN with dimensionality reduction using kNN-ISOMAP")

k_values = range(2, 20)
best_k = None
best_stress = float('inf')
for k in k_values:
    knn_graph = kneighbors_graph(X_normalized, n_neighbors=k, mode='distance', include_self=False)
    n_components, _ = connected_components(knn_graph, directed=False)
    if n_components > 1:
        continue
    
    isomap = Isomap(n_neighbors=k, n_components=2)
    embedding = isomap.fit_transform(X_normalized)
    geodesic_distances = isomap.dist_matrix_
    embedding_distances = squareform(pdist(embedding, metric='euclidean'))
    stress = mean_squared_error(geodesic_distances.ravel(), embedding_distances.ravel())
    
    if stress < best_stress:
        best_k = k
        best_stress = stress

print(f"Best k: {best_k}, Stress: {best_stress}")

isomap = Isomap(n_neighbors=k, n_components=2)
X_isomap = isomap.fit_transform(X_normalized)

X_train5, X_test5, y_train5, y_test5 = train_test_split(X_isomap,y, test_size=0.2, random_state=6740)

knn_isomap = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1, 30, 2))}


grid_search_isomap = GridSearchCV(
    knn_pca,
    param_grid,
    scoring=scoring,
    refit='balanced_accuracy',
    cv=10,
    n_jobs=-1
)

grid_search_isomap.fit(X_train5, y_train5)

print("Best K using Grid Search and ISOMAP:", grid_search_isomap.best_params_)
print(f"Balanced Accuracy using Grid Search and ISOMAP for K = {grid_search_isomap.best_params_}: {grid_search_isomap.best_score_}")

best_knn_isomap = grid_search_isomap.best_estimator_
y_pred_isomap = best_knn_isomap.predict(X_test5)

test_balanced_accuracy_isomap = balanced_accuracy_score(y_test5, y_pred_isomap)
test_fbeta_score_isomap = fbeta_score(y_test5, y_pred_isomap, beta=2, average='weighted')

print(f"Test Balanced Accuracy: {test_balanced_accuracy_isomap:.2f}")
print(f"Test F2 Score: {test_fbeta_score_isomap:.2f}")

plt.figure()
scatter = plt.scatter(X_test5[:,1], X_test5[:,1], c=y_test5, cmap='viridis', marker='o', label='True Labels', alpha=0.5)
plt.scatter(X_test5[:, 0], X_test5[:, 1], c=y_pred_isomap, marker='x', label='Predicted Labels', alpha=0.7, edgecolor='red')
plt.title("kNN Predictions vs True Labels_ISOMAP")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.colorbar(scatter, label='True Labels')
plt.savefig("kNN Predictions vs True Labels_ISOMAP.png")

cm = confusion_matrix(y_test5, y_pred_isomap)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix ISOMAP")
plt.savefig("confusion matrix_isomap.png")


plt.show()