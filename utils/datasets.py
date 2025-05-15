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

def load_datasets(dataset_path):
    df_train = pd.read_csv(dataset_path)
    X = df_train.iloc[:,:-1].values
    Y = df_train.iloc[:,-1:].values
    return X, Y


def get_dataset_path():  
    args = args_parser()
    # datasets = os.path.join(os.path.dirname('fed_lr'), 'traindata.csv')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_path = os.path.join(base_dir, '../datasets/', args.dataset)
    datasets_path = os.path.abspath(datasets_path)
    # print(datasets_path)
    return datasets_path



def save_model_weights(model_weights):

    args = args_parser()
    model_name = args.model
    today = datetime.date.today()
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    date4hash = datetime.datetime.now()
    hash_object = hashlib.md5(str(date4hash).encode())
    hash_str = str(hash_object.hexdigest()[:6])
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '../models/', model_name + "_" + str(today) + "_" + now_time +  "_" +  hash_str +  ".pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_weights, f)
        
# 数据划分 根据client_num 预先划分成多组数据
def get_data_loaders_split(batch_size=64, data_dir='', client_num=1):
    transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),  
                             (0.24205776, 0.23828046, 0.25874835)),  
    ])
    if data_dir == '':
        raise ValueError('data_dir must be specified')
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test') 
    train_set = datasets.ImageFolder(train_path, transform=transform)
    test_set = datasets.ImageFolder(test_path, transform=transform)
    num_train = len(train_set)
    client_loaders = []
    for i in range(client_num):
        indices = np.random.choice(num_train, num_train // 20, replace=False)  # 选择数据子集 抽取5%数据
        subset = Subset(train_set, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader

# 数据划分 根据client_num 预先划分成多组数据(cifar-10是各个类是乱序的 所以重新写只针对它的函数)
def get_cifar_loaders_split(batch_size=64, data_dir='', client_num=1):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 均值
                             (0.2023, 0.1994, 0.2010))   # CIFAR-10 标准差
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    num_train = len(train_set)
    client_loaders = []
    for i in range(client_num):
        indices = np.random.choice(num_train, num_train // 2, replace=False)  # 每个客户端抽50%
        subset = Subset(train_set, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader

# 客户端 加载分割数据
def acquire_and_load_data(dataset_name, max_clients, batch_size, data_dir='../data_split'):
    for i in range(1, max_clients + 1):
        base_name = f"{dataset_name}_client{i}_train.pt"
        full_path = os.path.join(data_dir, base_name)
        locked_path = os.path.join(data_dir, f"+{base_name}")
        if os.path.exists(full_path):
            try:
                os.rename(full_path, locked_path)
                client_id = i
                # print(f"[Client] 使用数据文件：{base_name}")
                train_data = torch.load(locked_path, weights_only=False)
                test_path = os.path.join(data_dir, f"{dataset_name}_test.pt")
                if not os.path.exists(test_path):
                    raise FileNotFoundError(f"测试集文件不存在: {test_path}")
                test_data = torch.load(test_path, weights_only=False)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                return client_id, train_loader, test_loader, locked_path, full_path
            except OSError:
                continue
    raise RuntimeError("未找到可用的训练数据文件，全部已被占用。")

# 服务器 对数据开展分割
def prepare_and_save_split(dataset_name, batch_size, raw_data_dir, save_dir, client_num):
    if dataset_name == 'cifar-10':
        client_data_loaders, test_loader = get_cifar_loaders_split(
        batch_size=batch_size,
        data_dir=raw_data_dir,
        client_num=client_num
    )
    else:
        client_data_loaders, test_loader = get_data_loaders_split(
            batch_size=batch_size,
            data_dir=raw_data_dir,
            client_num=client_num
        )
    os.makedirs(save_dir, exist_ok=True)
    for i, loader in enumerate(client_data_loaders):
        data = list(loader.dataset)
        save_path = os.path.join(save_dir, f"{dataset_name}_client{i+1}_train.pt")
        torch.save(data, save_path)
        # print(f"[Server] Saved train split to {save_path}")
    test_data = list(test_loader.dataset)
    test_save_path = os.path.join(save_dir, f"{dataset_name}_test.pt")
    torch.save(test_data, test_save_path)
    # print(f"[Server] Saved test set to {test_save_path}")