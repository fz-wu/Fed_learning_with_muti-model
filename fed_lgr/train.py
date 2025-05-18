# logistic_regression_model.py - 联邦学习客户端多分类逻辑回归版本（Softmax）

import socket
import pickle
from venv import logger
import numpy as np
import torch
from utils.datasets import save_model_weights, load_lgr_datasets, get_dataset_path
from utils.options import args_parser
from utils.datasets import load_csv, split_data

args = args_parser()

# 多分类逻辑回归模型类（使用Softmax）
class LogisticRegressionModel:
    def __init__(self, input_dim, num_classes, learning_rate=0.01, iterations=10):
        self.weights = np.zeros((input_dim, num_classes))
        self.bias = np.zeros((1, num_classes))
        self.lr = learning_rate
        self.epochs = iterations

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def loss(self, X, y):
        m = X.shape[0]
        z = X @ self.weights + self.bias
        probs = self.softmax(z)
        log_probs = -np.log(probs[np.arange(m), y.flatten().astype(int)] + 1e-8)
        cost = np.mean(log_probs)
        return cost

    def fit(self, X, y):
        m = X.shape[0]
        unique_labels = np.unique(y)
        label_map = {v: i for i, v in enumerate(unique_labels)}
        y_mapped = np.vectorize(label_map.get)(y.flatten())
        y_onehot = np.eye(self.weights.shape[1])[y_mapped]
        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            probs = self.softmax(z)
            dz = probs - y_onehot
            dw = (1 / m) * X.T @ dz
            db = (1 / m) * np.sum(dz, axis=0, keepdims=True)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self.weights, self.bias

    def evaluate(self, X, y):
        probs = self.softmax(X @ self.weights + self.bias)
        preds = np.argmax(probs, axis=1)
        accuracy = (preds == y.flatten()).mean() * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

# 本地训练与通信
def lgr_train():
    # 加载并划分数据
    X_train, X_test, y_train, y_test, dim, label_num = split_data()
    # print(f'label_num,{label_num}') # 这个相当于在加载数据时看一下 label的个数
    assert label_num == args.label_num # 如果希望每轮按照输入的label_num设定分组数 则放开这个
    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
    X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-8)

    model = LogisticRegressionModel(input_dim=dim, num_classes=label_num,
                                    learning_rate=args.lr, iterations=args.epochs)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.port))

    for round in range(args.round):
        print(f"\nRound {round + 1}")
        logger.info(f"Round {round + 1}")
        # 本地训练
        weights, bias = model.fit(X_train, y_train)
        model.evaluate(X_test, y_test)

        # 打包并发送本地模型参数
        payload = {
            'weights': (weights, bias),
            'num_samples': len(X_train)
        }
        serialized = pickle.dumps(payload)
        client_socket.sendall(len(serialized).to_bytes(4, 'big'))
        client_socket.sendall(serialized)
        print("Sent weights and sample count to server.")
        logger.info("Sent weights and sample count to server.")
        # 接收聚合模型
        try:
            length_data = client_socket.recv(4)
            total_length = int.from_bytes(length_data, 'big')
            serialized_model = client_socket.recv(total_length)
            w_new, b_new = pickle.loads(serialized_model)
            model.weights = w_new
            model.bias = b_new
            print("Updated model from server.")
            logger.info("Updated model from server.")
        except Exception as e:
            print(f"Error receiving updated model: {e}")
            logger.error(f"Error receiving updated model: {e}")
            break

    save_model_weights((model.weights, model.bias))
    client_socket.close()
