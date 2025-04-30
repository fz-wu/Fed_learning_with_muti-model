import pandas as pd
import os

def load_datasets(dataset_path):
    df_train = pd.read_csv(dataset_path)
    X = df_train.iloc[:,:-1].values
    Y = df_train.iloc[:,-1:].values
    return X, Y