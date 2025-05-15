import socket
import threading
import pickle
import numpy as np
from time import sleep
from utils.options import args_parser
from fed_lgr.server import Server
from fed_lgr.model import LogisticRegressionModel
from fed_lgr.heart_disease_dataset import get_data, get_labels
import torch
# 全局变量
args = args_parser()
client_weights = []
client_sockets = []  # 存储所有客户端socket
client_count = 0
client_lock = threading.Lock()
all_clients_connected_event = threading.Event()
NUM_ROUNDS =  args.round # 联邦学习迭代轮数


def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError('Socket closed before receiving all data')
        data += more
    return data

def create_connect(client_num, port):
    global client_weights, client_count, all_clients_connected_event, client_sockets
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print("client number = {}".format(client_num))
    print('Waiting for connection...')

    for _ in range(client_num):
        client_socket, client_addr = server_socket.accept()
        client_sockets.append(client_socket)
        print("client_addr: {}".format(client_addr))

    for round in range(NUM_ROUNDS):
        print(f"Starting aggregation round {round + 1}...")
        sleep(1)
        print("All clients connected. Starting aggregation...")
        # 接收所有权重
        recved_weights = []
        for client_socket in client_sockets:
            serialized_clinet_weight = client_socket.recv(102400)
            if not serialized_clinet_weight:
                print("Warning: received empty data from client", client_socket.getpeername())
                continue
            client_weight = pickle.loads(serialized_clinet_weight)
            print("Received weight from client {}: {}".format(client_socket.getpeername(),client_weights))
            recved_weights.append(client_weight) # 等待客户端发送权重

        print("recved_client_weights:{}".format(recved_weights))
        if args.model == "lr":
            aggregated_weights = aggregate_lr(recved_weights)
        elif args.model == "lgr":
            aggregated_weights = aggregate_lgr(recved_weights)
        elif args.model == "kmeans":
            aggregated_weights = aggregate_kmeans(recved_weights)
        elif args.model == "svm":
            aggregated_weights = aggregate_svm(recved_weights)
        elif args.model == "cnn":
            aggregated_weights = aggregate_cnn(recved_weights)
        
        print("Aggregated weights:", aggregated_weights)

        # 发送聚合后的权重到所有客户端
        for client_socket in client_sockets:
            try:
                serialized_weights = pickle.dumps(aggregated_weights)
                client_socket.sendall(serialized_weights)
                print("Sent aggregated weights to client {}.".format(client_socket.getpeername()))
            except Exception as e:
                print("Error sending weights to client:", e)

        print("Weights sent to all clients. Ready for the next round.")
    print("All rounds completed. Closing server socket.")

def create_connect_cnn(client_num, port):
    global client_sockets
    client_sockets = []

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print(f"client number = {client_num}")
    print('Waiting for connection...')

    for _ in range(client_num):
        client_socket, client_addr = server_socket.accept()
        client_sockets.append(client_socket)
        print(f"client_addr: {client_addr}")

    for round in range(NUM_ROUNDS):
        print(f"\nStarting aggregation round {round + 1}...")
        sleep(1)
        recved_payloads = []
        for client_socket in client_sockets:
            try:
                # 从 socket 中接收前 4 个字节，表示后续数据的总长度（以字节为单位）
                length_data = recvall(client_socket, 4)
                # 将收到的 4 个字节转换为整数，得到实际要接收的数据长度（大端字节序）
                total_length = int.from_bytes(length_data, byteorder='big')
                # 持续从 socket 中接收数据，直到接收到 total_length 字节为止，确保数据完整
                serialized_data = recvall(client_socket, total_length)
                payload = pickle.loads(serialized_data)
                client_weight = payload['weights']
                sample_num = payload['num_samples']
                recved_payloads.append((client_weight, sample_num))
                print(f"Received weight and sample count ({sample_num}) from client {client_socket.getpeername()}.")
            except Exception as e:
                print(f"Error receiving data from client {client_socket.getpeername()}: {e}")
        # 聚合 [(weight, sample_num), ...]
        aggregated_weights = aggregate_cnn(recved_payloads)
        for client_socket in client_sockets:
            try:
                serialized_weights = pickle.dumps(aggregated_weights)
                data_length = len(serialized_weights)
                client_socket.sendall(data_length.to_bytes(4, byteorder='big'))
                client_socket.sendall(serialized_weights)
                print(f"Sent aggregated weights to client {client_socket.getpeername()}.")
            except Exception as e:
                print(f"Error sending weights to client {client_socket.getpeername()}: {e}")
        print("Weights sent to all clients. Ready for the next round.")
    print("All rounds completed. Closing server socket.")
    server_socket.close()
    for sock in client_sockets:
        sock.close()


# SVM 服务端端口监听并聚合客户端上传模型的函数
def create_connect_svm(client_num, port):
    client_sockets = []

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print(f"client number = {client_num}")
    print('Waiting for connection...')

    for _ in range(client_num):
        client_socket, client_addr = server_socket.accept()
        client_sockets.append(client_socket)
        print(f"client_addr: {client_addr}")

    for round in range(NUM_ROUNDS):
        print(f"\nStarting aggregation round {round + 1}...")
        sleep(1)
        recved_payloads = []
        for client_socket in client_sockets:
            try:
                length_data = recvall(client_socket, 4)
                total_length = int.from_bytes(length_data, byteorder='big')
                serialized_data = recvall(client_socket, total_length)
                payload = pickle.loads(serialized_data)
                client_weight = payload['weights']
                sample_num = payload['num_samples']
                recved_payloads.append((client_weight, sample_num))
                print(f"Received weights and sample count ({sample_num}) from client {client_socket.getpeername()}.")
            except Exception as e:
                print(f"Error receiving data from client {client_socket.getpeername()}: {e}")

        aggregated_weights = aggregate_svm(recved_payloads)

        for client_socket in client_sockets:
            try:
                serialized_weights = pickle.dumps(aggregated_weights)
                data_length = len(serialized_weights)
                client_socket.sendall(data_length.to_bytes(4, byteorder='big'))
                client_socket.sendall(serialized_weights)
                print(f"Sent aggregated weights to client {client_socket.getpeername()}.")
            except Exception as e:
                print(f"Error sending weights to client {client_socket.getpeername()}: {e}")

        print("Weights sent to all clients. Ready for the next round.")

    print("All rounds completed. Closing server socket.")
    server_socket.close()
    for sock in client_sockets:
        sock.close()


def create_lgr_connect(client_num, port, round_num):
    global client_sockets
    x1, x2, x3 = get_data()
    y = get_labels().values.reshape(-1, 1)
    server = Server(0.0001, LogisticRegressionModel, data=(x1, y))
    N = x1.shape[0]

def aggregate_lr(weights):

    # print("wait weights:{}".format(weights))
    total_W = np.zeros_like(weights[0][0])
    total_b = 0
    for W, b in weights:
        total_W += W
        total_b += b
    aggregated_W = total_W / len(weights)
    aggregated_b = total_b / len(weights)
    return (aggregated_W, aggregated_b)

def aggregate_kmeans(weights):
    weights = np.array(weights)  # shape: (num_clients, n_clusters, n_features)
    print("weights:{}".format(weights))
    aggregated_centers = np.mean(weights, axis=0)  # shape: (n_clusters, n_features)
    return aggregated_centers

def aggregate_lgr(weights):
    pass

# 联邦聚合函数：按客户端样本数加权平均多个权重
def aggregate_svm(payloads):
    total_samples = sum(n for _, n in payloads)
    label_num, dim = payloads[0][0].shape
    aggregated = np.zeros((label_num, dim))
    for weights, num_samples in payloads:
        aggregated += weights * (num_samples / total_samples)
    return aggregated

# cnn模型聚合方法
def aggregate_cnn(client_payloads):
    aggregated_weights = {}
    total_samples = sum(num_samples for _, num_samples in client_payloads)
    # 取第一个客户端的结构作为参考
    for layer_name in client_payloads[0][0].keys():
        # 初始化聚合张量
        layer_sum = torch.zeros_like(client_payloads[0][0][layer_name])
        for state_dict, num_samples in client_payloads:
            weight = state_dict[layer_name].float()  # 确保计算时是 float
            layer_sum += weight * (num_samples / total_samples)
        aggregated_weights[layer_name] = layer_sum
    return aggregated_weights

def send_weights(target_host, port, weights):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((target_host, port))
    serialized_weights = pickle.dumps(weights)
    sock.sendall(serialized_weights)
    new_weight = sock.recv(102400)
    new_weight = pickle.loads(new_weight)
    print("client_weight:{}".format(new_weight))
    # sock.close()
    return new_weight
