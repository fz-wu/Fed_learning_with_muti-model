import pandas as pd
import os
from utils.options import args_parser
import pickle
import hashlib
import datetime
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import ImageFolder

def load_datasets(dataset_path):
    df_train = pd.read_csv(dataset_path)
    X = df_train.iloc[:,:-1].values
    Y = df_train.iloc[:,-1:].values
    return X, Y


def get_dataset_path():  
    args = args_parser()
    # datasets = os.path.join(os.path.dirname('fed_lr'), 'traindata.csv')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(base_dir)
    print(args.dataset)
    datasets_path = os.path.join(base_dir, '../datasets/', args.dataset)
    datasets_path = os.path.abspath(datasets_path)
    # print(datasets_path)
    return datasets_path

def save_model_weights(model_weights):
    args = args_parser()
    model_name = args.model
    today = datetime.date.today()
    now_time = datetime.datetime.now().strftime("%H-%M-%S")
    date4hash = datetime.datetime.now()
    hash_object = hashlib.md5(str(date4hash).encode())
    hash_str = str(hash_object.hexdigest()[:6])
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '../models/', model_name + "_" + str(today) + "_" + now_time +  "_" +  hash_str +  ".pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_weights, f)

# cnn 数据加载
def load_datasets_cnn(model, dataset, batch_size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, f'../datasets/{model}/{dataset}'))
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),  
                             (0.24205776, 0.23828046, 0.25874835)),  
    ])
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 加载 CSV 数据
def load_csv(filepath, minmax=None, standardize=True, bias_term=True):
    # 自动判断分隔符是 ',' 还是 ';'
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','
    df = pd.read_csv(filepath, sep=sep)
    labels = df.iloc[:, -1].values  # 使用最后一列作为标签
    features = df.iloc[:, :-1].values
    if minmax is not None:
        scaler = MinMaxScaler(feature_range=minmax)
        features = scaler.fit_transform(features)
    elif standardize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    if bias_term:
        features = np.hstack([np.ones((features.shape[0], 1)), features])
    return features, labels

# 数据集划分函数
def split_data(test_size=0.3):
    dataset_path = get_dataset_path()+'.csv'
    X, y = load_csv(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
    return X_train, X_test, y_train, y_test, X.shape[1], len(np.unique(y))

def load_lgr_datasets(dataset_path, client_id=None, total_clients=1):
    df_train = pd.read_csv(dataset_path)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱

    if client_id is not None:
        # 划分数据为多个客户端子集
        subsets = np.array_split(df_train, total_clients)
        df_train = subsets[client_id]

    X = df_train.iloc[:, :-1].values
    Y = df_train.iloc[:, -1:].values
    return X, Y

