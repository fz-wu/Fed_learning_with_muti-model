from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 数据标准化
from sklearn.preprocessing import StandardScaler

import numpy as np
import sys, os
from .kmeans_class import FedKMeans
from .clustering.kmeans_python import KMeansFederated
from .kmeans_class import KMeans
from utils.options import args_parser
from utils.datasets import save_model_weights
# from utils.datasets import load_datasets, get_dataset_path

def init_kmeans_sklearn(n_clusters, batch_size, seed, init_centroids='random'):
    # init kmeans
    if batch_size is not None:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init=init_centroids,  # 'random', 'k-means++', ndarray (n_clusters, n_features)
            max_iter=9,
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


def kmeans_train():
    args = args_parser()
    iris = load_iris()
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target,test_size=0.2)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    kmeans = FedKMeans(n_clusters=args.label_num, random_state=1, max_iter=args.round).fit(x_train)
    print(x_train.shape)
    # print(kmeans.labels_)
    # is_same = zip(kmeans.labels_, y_train)
    # out = []
    # for i, j in is_same:
    #     if i == j:
    #         out.append(1)
    #     else:
    #         out.append(0)  
    # print(out)
    # acc = sum(out) / len(out)
    # print("acc: ", acc)
        # kmeans.cluster_centers_

    print(y_train)
    # print(kmeans.predict([[0, 0], [12, 3]]))
    print(kmeans.cluster_centers_)
    save_model_weights(kmeans.cluster_centers_,)
    # x_train = [x_train[i:i+1] for i in range(x_train.shape[0])]
    # kmeans = KMeansFederated(
    # n_clusters=3,
    # sample_fraction=0,
    # verbose=True,
    # learning_rate=0.5,
    # adaptive_lr=0.1,
    # max_iter=100,
    # momentum=0.8,
    # ).fit(X=x_train)
    # print(kmeans.cluster_centers_)

    # centroids, overall_counts = kmeans.fit(X=x_train)
if __name__ == "__main__":
    # test_sk()
    kmeans_train()
