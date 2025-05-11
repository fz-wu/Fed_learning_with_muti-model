import pandas as pd
import os
from utils.options import args_parser
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
