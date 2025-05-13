import socket
import threading
import pickle
import numpy as np
from time import sleep
from utils.options import args_parser
from fed_lgr.server import Server
from fed_lgr.model import LogisticRegressionModel
from fed_lgr.heart_disease_dataset import get_data, get_labels
# 全局变量
args = args_parser()
client_weights = []
client_sockets = []  # 存储所有客户端socket
client_count = 0
client_lock = threading.Lock()
all_clients_connected_event = threading.Event()
NUM_ROUNDS =  args.round # 联邦学习迭代轮数

def create_connect(client_num, port):
    global client_weights, client_count, all_clients_connected_event, client_sockets
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print("client number = {}".format(client_num))
    print('Waiting for connection...')

    for i in range(client_num):
        client_socket, client_addr = server_socket.accept()
        client_sockets.append(client_socket)
        print("client_addr: {}".format(client_addr))
        if args.model == "lgr":
            client_socket.sendall(pickle.dumps(i + 1))
    for round in range(NUM_ROUNDS):
        print(f"Starting aggregation round {round + 1}...")
        sleep(1)
        print("All clients connected. Starting aggregation...")
        if args.model == "lgr":
            # 处理 LGR 模型逻辑回归
            recved_weights = []
            for client_socket in client_sockets:
                data = client_socket.recv(102400)
                ids, z = pickle.loads(data)
                print("Received z from client {}: ids shape {}, z shape {}".format(
                    client_socket.getpeername(), np.shape(ids), np.shape(z)))
                recved_weights.append((ids, z))
        else:
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

def aggregate_lgr(client_data_list):
    """
    联邦逻辑回归的服务端聚合逻辑。
    参数 client_data_list: [(ids, z_i), (ids, z_j), ...]
    """
    x1, _, _ = get_data()
    y = get_labels().values.reshape(-1, 1)
    server = Server(0.0001, LogisticRegressionModel, data=(x1, y))

    z_parts = []
    ids_list = []

    for ids, z_i in client_data_list:
        z_parts.append(z_i)
        ids_list = ids  # 所有客户端应该是同一批 id

    server.forward(ids_list)

    for z_i in z_parts:
        server.receive(z_i, 0)

    server.compute_gradient()
    diff = server.send()
    server.update_model()

    return diff  # 返回给客户端

def aggregate_svm(weights):
    pass

def aggregate_cnn(weights):
    pass

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


