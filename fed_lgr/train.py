from utils.datasets import load_lgr_datasets, get_dataset_path, save_model_weights
import pickle
import socket
import numpy as np
from utils.options import args_parser
from fed_lgr.model import LogisticRegressionModel
args = args_parser()


class LogisticRegression:
    def __init__(self, data, learning_rate=0.01, iterations=10):
        self.X = data[0]
        self.y = data[1]
        self.X = (self.X - self.X.mean(axis=0)) / (self.X.std(axis=0) + 1e-8)
        self.weights = np.zeros((self.X.shape[1], 1))  # 初始化权重
        self.bias = 0
        self.learning_rate = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid 函数

    def loss(self):
        m = self.y.shape[0]
        cost = -(1 / m) * np.sum(self.y * np.log(self.sigmoid(self.X.dot(self.weights) + self.bias)) +
                                 (1 - self.y) * np.log(1 - self.sigmoid(self.X.dot(self.weights) + self.bias)))
        return cost

    def fit(self):
        m = self.X.shape[0]
        for i in range(self.iterations):
            z = np.dot(self.X, self.weights) + self.bias
            prediction = self.sigmoid(z)
            dw = (1 / m) * np.dot(self.X.T, (prediction - self.y))  # 计算梯度
            db = (1 / m) * np.sum(prediction - self.y)
            self.weights -= self.learning_rate * dw  # 更新权重
            self.bias -= self.learning_rate * db  # 更新偏置
        return self.weights, self.bias  # 返回更新后的权重和偏置


def lgr_train():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.port))

    # 接收自己的 client_id
    client_id_data = client_socket.recv(1024)
    client_id = pickle.loads(client_id_data)
    datasets_path = get_dataset_path()
    X, Y = load_lgr_datasets(datasets_path, client_id=client_id, total_clients=args.client_num)
    # model = LogisticRegression(data=(X, Y), learning_rate=args.lr, iterations=10)
    model = LogisticRegressionModel(X, Y, lr=args.lr, label_num=args.label_num)
    print(f"[Client] Assigned client_id = {client_id}")
    initial_weights_data = client_socket.recv(10240)
    initial_theta = pickle.loads(initial_weights_data)
    model.w, model.b = initial_theta
    print(f"[Client] Received initial weights from server.")

    for r in range(args.round):
        print(f"[Client] Round {r}")

        # Step 1: 从第1轮起接收服务端模型参数
        if r > 0:
            weights = client_socket.recv(10240)
            if not weights:
                print("[Client] No weights received from server.")
                break
            updated_theta = pickle.loads(weights)
            model.w, model.b = updated_theta
            print(f"[Client] Received updated weights and bias from server.")

        # Step 2: 每一轮都计算 z_i 并发送
        z_i = np.dot(model.X, model.w) + model.b
        sample_ids = np.arange(client_id * len(model.X), (client_id + 1) * len(model.X))
        client_socket.sendall(pickle.dumps((sample_ids, z_i)))
        print(f"[Client] Sent z_i and sample_ids to server.")

        # 最后一轮也接收一次用于保存
    final_weights = client_socket.recv(10240)
    if final_weights:
        final_theta = pickle.loads(final_weights)
        model.weights, model.bias = final_theta
        print(f"[Client] Final weights received. Saving model.")
        save_model_weights(final_theta)
    # 保存模型权重
    client_socket.close()
