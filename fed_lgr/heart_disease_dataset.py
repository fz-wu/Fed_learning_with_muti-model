#This script is used to split the dataset into three partitions vertically
import sklearn
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.options import args_parser
args = args_parser()
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, 'heart_disease.csv'))


y = df["output"]
X = df.drop(["output"],axis=1)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

x1 = x_train.iloc[:, :4].values
x2 = x_train.iloc[:, 4:9].values
x3 = x_train.iloc[:, 9:].values

x1_test = x_test.iloc[:, :4].values
x2_test = x_test.iloc[:, 4:9].values
x3_test = x_test.iloc[:, 9:].values


def get_data():
    return x1,x2,x3
# def get_dataset_path():
#     # 保证始终从项目根目录定位文件
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     root_dir = os.path.abspath(os.path.join(base_dir, '..'))  # 回到 fedlearning/
#     dataset_path = os.path.join(root_dir, args.dataset)
#     return dataset_path
# def get_data():
#     df = pd.read_csv(get_dataset_path())
#     X = df.drop(columns=['output']).values
#     num_clients = args.client_num
#     num_features = X.shape[1]
#     step = num_features // num_clients
#
#     feature_splits = []
#     for i in range(num_clients):
#         start = i * step
#         end = (i + 1) * step if i < num_clients - 1 else num_features
#         feature_splits.append(X[:, start:end])
#
#     return feature_splits  # 返回 list：[x1, x2, x3, ...]
def get_labels():
    return y_train

def get_testdata():
    return x1_test,x2_test,x3_test

def get_testlabels():
    return y_test

def get_dataset():
    return x_train, x_test
