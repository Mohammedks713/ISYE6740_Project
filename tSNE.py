import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
tsne = TSNE(n_components=2, random_state=6740)
X_tsne = tsne.fit_transform(X_normalized)

plt.figure()
for i in np.unique(y):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=f'Class {i}')
plt.legend()
plt.title("2D t-SNE Clustering of MRI Scans")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig("2D t-SNE Clustering of MRI Scans.png")
plt.show()