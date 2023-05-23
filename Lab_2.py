import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis

df = pd.read_csv('glass.csv')
var_names = list(df.columns)
labels = df.to_numpy('int')[:, -1]
data = df.to_numpy('float')[:, :-1]
     
data = preprocessing.minmax_scale(data)

fig, axs = plt.subplots(2, 4)
fig.tight_layout()
fig.set_figheight(7)
fig.set_figwidth(15)

for i in range(data.shape[1] - 1):
    scatter = axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=labels, cmap='hsv')
    legend = axs[i // 4, i % 4].legend(*scatter.legend_elements(),
                                       loc="upper right", title="Classes")
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit(data).transform(data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='hsv')
plt.show()
pca = PCA(n_components=4)
pca_data = pca.fit(data).transform(data)

print(sum(pca.explained_variance_ratio_))

pca = pca.inverse_transform(pca_data)

parameters = ('auto', 'full', 'arpack', 'randomized')
for parameter in parameters:
    print(f'Параметр: {parameter}')
    pca = PCA(n_components=4, svd_solver=parameter)
    pca_data = pca.fit(data).transform(data)
    print(pca_data)
    print('*' * 50)

kernel_pca = KernelPCA(n_components=2)
kernel_pca_data = kernel_pca.fit(data).transform(data)
kernel_pca_data
plt.scatter(kernel_pca_data[:, 0], kernel_pca_data[:, 1], c=labels, cmap='hsv')
plt.show()
parameters = ('linear', 'poly', 'rbf', 'sigmoid', 'cosine')

for parameter in parameters:
    print(f'Параметр: {parameter}')
    kernel_pca = KernelPCA(n_components=4, kernel=parameter)
    kernel_pca_data = kernel_pca.fit(data).transform(data)
    print(kernel_pca_data)
    print('*' * 50)
    
kernel_pca = KernelPCA(n_components=2, kernel='linear')
kernel_pca_data = kernel_pca.fit(data).transform(data)
plt.scatter(kernel_pca_data[:, 0], kernel_pca_data[:, 1], c=labels, cmap='hsv')
plt.show()

sparse_pca = SparsePCA(n_components=2, alpha=1)
sparse_pca_data = sparse_pca.fit(data).transform(data)
sparse_pca_data

plt.scatter(sparse_pca_data[:, 0], sparse_pca_data[:, 1], c=labels, cmap='hsv')
plt.show()

parameters = ('lars', 'cd')

for parameter in parameters:
    print(f'Параметр: {parameter}')
    sparse_pca = SparsePCA(n_components=4, alpha=0.01, method=parameter)
    sparse_pca_data = sparse_pca.fit(data).transform(data)
    print(sparse_pca_data)
    print('*' * 50)
     
sparse_pca = SparsePCA(n_components=2, alpha=0)
sparse_pca_data = sparse_pca.fit(data).transform(data)

plt.scatter(sparse_pca_data[:, 0], sparse_pca_data[:, 1], c=labels, cmap='hsv')
plt.show()

pca = FactorAnalysis(n_components = 2)
pca_data = pca.fit(data).transform(data)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv')
plt.show()
