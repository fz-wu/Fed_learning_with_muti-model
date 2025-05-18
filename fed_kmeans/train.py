# from json import load
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# 数据标准化
from sklearn.preprocessing import StandardScaler
import logging
# import numpy as np
from .kmeans_class import FedKMeans
from utils.options import args_parser
from utils.datasets import save_model_weights,get_log_path, load_datasets, get_dataset_path


logging.basicConfig(level=logging.INFO, filename=get_log_path(),format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kmeans_train():
    args = args_parser()
    # iris = load_iris()
    dataset = get_dataset_path()
    X, Y = load_datasets(dataset)
    # x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target,test_size=0.2)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(X)
    # x_test = transfer.transform(x_test)
    kmeans = FedKMeans(n_clusters=args.label_num, random_state=1, max_iter=args.round).fit(x_train)
    # print(X.shape)
    print("All round already finished.")
    logger.info("All round already finished.")
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

    # print(kmeans.predict([[0, 0], [12, 3]]))
    print(kmeans.cluster_centers_)
    save_model_weights(kmeans.cluster_centers_,)
    print("save model weights success.")
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
