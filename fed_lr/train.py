import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from dotenv import find_dotenv, load_dotenv
import os
import sys
import socket
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from server import server
import dataset as dd
# Local Data for Each Clients


class Model():
    
    def __init__( self, data, learning_rate=0.001, iterations=10) :
        self.x = data[0]
        self.y = data[1]
        # self.w = theta[0]
        self.w =  np.zeros((self.x.shape[1],1))
        self. b = 0
        self.learning_rate = learning_rate

        self.epochs = iterations
        
    def loss(self):
        cost = np.sum((((self.x.dot(self.w) + self.b) - self.y) ** 2) / (2*len(self.y)))
        return cost

    def fit(self,theta):
        self.w = theta[0]
        self.b = theta[1]
        cost_list = [0] * self.epochs
    
        for epoch in range(self.epochs):
            z = self.x.dot(self.w) + self.b
            loss = z - self.y
            
            weight_gradient = self.x.T.dot(loss) / len(self.y)
            bias_gradient = np.sum(loss) / len(self.y)
            
            self.w = self.w - self.learning_rate*weight_gradient
            self.b = self.b - self.learning_rate*bias_gradient
    
            cost = self.loss()
            cost_list[epoch] = cost
            
            # if (epoch%(self.epochs/10)==0):
            #     print("Cost is:",cost)
            
        return self.w, self.b

class Participant():
    def __init__(self,model:Model,data):
        self.data = data
        self.model = model(data=self.data)
        

    def receive_from_server(self,theta):
        self.theta = theta

    def train(self):
        self.theta = self.model.fit(self.theta)
        

    def send_to_server(self):
        return self.theta



def predict(X,theta):
    return np.dot(X,theta[0]) + theta[1]

def lr_train(X, Y, host='127.0.0.1', port=10000, iterations=10):
    """
    使用梯度下降法训练线性回归模型，并通过 socket 与服务器通信。

    Args:
        X: 特征矩阵 (NumPy 数组)。
        Y: 目标变量 (NumPy 数组)。
        host: 服务器 IP 地址。
        port: 服务器端口。
        iterations: 迭代次数。

    Returns:
        训练后的模型参数 (元组，包含权重和偏置)。
    """
    M, N = X.shape
    # 初始化模型参数
    w = np.zeros((N, 1))  # 权重初始化为 0
    b = 0  # 偏置初始化为 0
    theta = (w, b)

    # 创建 Model 实例
    model = Model(data=(X, Y), learning_rate=0.001, iterations=iterations)
    model.w = w
    model.b = b

    try:
        # 创建 socket 连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # 循环进行训练和参数交换
        for epoch in range(iterations):
            # 1. 接收服务器发送的参数
            serialized_theta = sock.recv(10240)
            if not serialized_theta:
                print("Server disconnected.")
                break
            theta = pickle.loads(serialized_theta)
            model.w = theta[0]
            model.b = theta[1]
            print("Received theta from server")

            # 2. 本地训练
            z = X.dot(model.w) + model.b
            loss = z - Y

            weight_gradient = X.T.dot(loss) / len(Y)
            bias_gradient = np.sum(loss) / len(Y)

            model.w = model.w - model.learning_rate * weight_gradient
            model.b = model.b - model.learning_rate * bias_gradient

            # 3. 将训练后的参数发送回服务器
            theta = (model.w, model.b)
            serialized_theta = pickle.dumps(theta)
            sock.sendall(serialized_theta)
            print("Sent theta to server")

        print("Training complete.")

    except Exception as e:
        print("Error in lr_train:", e)
    finally:
        sock.close()

    return theta