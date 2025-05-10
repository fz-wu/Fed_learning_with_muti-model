from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 数据标准化
from sklearn.preprocessing import StandardScaler

import numpy as np
import sys, os

# from utils.datasets import load_datasets, get_dataset_path

def init_kmeans_sklearn(n_clusters, batch_size, seed, init_centroids='random'):
    # init kmeans
    if batch_size is not None:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init=init_centroids,  # 'random', 'k-means++', ndarray (n_clusters, n_features)
            max_iter=100,
            tol=0,  # 0.0001 (if not zero, adds compute overhead)
            n_init=1,
            # verbose=True,
            batch_size=batch_size,
            compute_labels=True,
            max_no_improvement=100,  # None
            init_size=None,
            reassignment_ratio=0.1 / n_clusters,
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init=init_centroids,  # 'random', 'k-means++', ndarray (n_clusters, n_features)
            max_iter=100,
            tol=0.001,
            n_init=1,
            # verbose=True,
            precompute_distances=True,
            algorithm='full',  # 'full',  # 'elkan',
        )
    return kmeans


def test_sk():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
    # print(X)
    print(X.shape)
    print(kmeans.labels_)
    print(kmeans.predict([[0, 0], [12, 3]]))
    print(kmeans.cluster_centers_)


def kmeans_train():
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target,test_size=0.2)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    kmeans = KMeans(n_clusters=3, random_state=1).fit(x_train)
    print(x_train.shape)
    print(kmeans.labels_)
    for i in y_train:
        if i == 0:
            i = 1
        elif i == 1:
            i = 0

    print(y_train)
    # print(kmeans.predict([[0, 0], [12, 3]]))
    # print(kmeans.cluster_centers_)

if __name__ == "__main__":
    # test_sk()
    test_train()
