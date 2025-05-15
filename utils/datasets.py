import pandas as pd
import os
from utils.options import args_parser
import pickle
import hashlib
import datetime
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