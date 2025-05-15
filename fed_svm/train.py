# svm_model.py - 多分类联邦学习版本（使用 One-vs-Rest 策略 + CSV 数据）

import numpy as np
import pandas as pd
import socket
import pickle
from time import sleep
from utils.transmit import recvall
from utils.datasets import save_model_weights
from utils.options import args_parser
from utils.datasets import load_csv, split_data

args = args_parser()

# SVM 模型封装类
class SVMClassifier:
    def __init__(self, label_num, dim):
        # 初始化模型，创建用于多分类的每类一个权重向量
        self.label_num = label_num
        self.dim = dim
        self.weights = np.zeros((label_num, dim))

    def grad(self, w, X, y):
        # 计算 hinge 损失下的梯度
        margin = y * np.dot(X, w)
        mask = margin < 1
        grad = -np.sum((y[mask][:, None]) * X[mask], axis=0)
        return grad

    def train_ovr(self, X, y, lr=0.01, epochs=100):
        # 使用 One-vs-Rest 策略训练多分类 SVM 模型
        for c in range(self.label_num):
            y_binary = np.where(y == c, 1, -1)  # 当前类为 1，其他类为 -1
            w = self.weights[c]
            for _ in range(epochs):
                grad = self.grad(w, X, y_binary)
                w -= lr * grad / len(X)  # 均值梯度下降更新
            self.weights[c] = w

    def predict(self, X):
        # 对每个样本计算各类别得分，取最大得分对应的类别作为预测
        scores = np.dot(X, self.weights.T)
        return np.argmax(scores, axis=1)

    def accuracy(self, X, y):
        # 计算预测准确率
        y_pred = self.predict(X)
        return (y_pred == y).mean() * 100

# 客户端训练流程
def svm_train():
    X_train, X_test, y_train, y_test, dim, label_num = split_data(f"./datasets/{args.dataset}.csv", label_col='target')
    # assert label_num == args.label_num
    model = SVMClassifier(label_num=label_num, dim=dim)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.port))

    for round in range(args.round):
        print(f"\n[Round {round + 1}]")
        model.train_ovr(X_train, y_train, lr=args.lr, epochs=args.epochs)
        acc = model.accuracy(X_test, y_test)
        print(f"Test Accuracy: {acc:.2f}%")

        payload = {
            'weights': model.weights,
            'num_samples': len(X_train)
        }

        serialized_payload = pickle.dumps(payload)
        client_socket.sendall(len(serialized_payload).to_bytes(4, byteorder='big'))
        client_socket.sendall(serialized_payload)
        print("Sent model weights and sample count to server.")

        try:
            length_data = recvall(client_socket, 4)
            total_length = int.from_bytes(length_data, byteorder='big')
            serialized_weights = recvall(client_socket, total_length)
            model.weights = pickle.loads(serialized_weights)
            print("Updated model from server.")
        except Exception as e:
            print(f"Error receiving updated weights: {e}")
            break

    save_model_weights(model.weights)
    client_socket.close()