import socket
import threading
import pickle
import numpy as np

# 全局变量
client_weights = []
client_sockets = []  # 存储所有客户端socket
client_count = 0
client_lock = threading.Lock()
all_clients_connected_event = threading.Event()
NUM_ROUNDS = 10  # 联邦学习迭代轮数

def tcplink(sock, addr, client_id, client_num):
    global client_weights, client_count

    print('Accept new connection from {}'.format(addr))
    try:
        for round in range(NUM_ROUNDS):
            # 1. 接收客户端发送的参数
            weight = sock.recv(10240)
            if not weight:
                print(f"Client {client_id} disconnected.")
                break
            weight = pickle.loads(weight)
            print(f"Round {round+1}: Received weight from {addr}: {type(weight)}")

            with client_lock:
                client_weights.append(weight)
                client_count += 1
                if client_count == client_num:
                    print("All clients connected. Proceeding to aggregation.")
                    all_clients_connected_event.set()

            # 等待主线程聚合并发送新权重
            agg_weight = sock.recv(10240)
            if not agg_weight:
                print(f"Client {client_id} disconnected after aggregation.")
                break
            agg_weight = pickle.loads(agg_weight)
            print(f"Round {round+1}: Client {client_id} received new weights.")

    except Exception as e:
        print("Error in tcplink:", e)
    finally:
        sock.close()
        print('Connection from %s:%s closed.' % addr)

def create_connect(client_num, port):
    global client_weights, client_count, all_clients_connected_event, client_sockets
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print("client number = {}".format(client_num))
    print('Waiting for connection...')

    threads = []
    client_id_counter = 0
    for _ in range(client_num):
        client_socket, client_addr = server_socket.accept()
        client_sockets.append(client_socket)
        print("client_addr: {}".format(client_addr))
        t = threading.Thread(target=tcplink, args=(client_socket, client_addr, client_id_counter, client_num))
        threads.append(t)
        t.start()
        print("Thread started for client: {}".format(client_addr))
        client_id_counter += 1

    for round in range(NUM_ROUNDS):
        print(f"Starting aggregation round {round + 1}...")
        all_clients_connected_event.wait()
        all_clients_connected_event.clear()

        print("All clients connected. Starting aggregation...")
        print("client_weights:{}".format(client_weights))
        aggregated_weights = aggregate_weights(client_weights)
        print("Aggregated weights:", aggregated_weights)

        with client_lock:
            client_weights_copy = client_weights[:]
            client_weights = []
            client_count = 0

        # 发送聚合后的权重到所有客户端
        for client_socket in client_sockets:
            try:
                serialized_weights = pickle.dumps(aggregated_weights)
                client_socket.sendall(serialized_weights)
                print("Sent aggregated weights to client {}.".format(client_socket.getpeername()))
            except Exception as e:
                print("Error sending weights to client:", e)

        print("Weights sent to all clients. Ready for the next round.")

def aggregate_weights(weights):
    """
    简单平均聚合
    """
    print("wait weights:{}".format(weights))
    total_W = np.zeros_like(weights[0][0])
    total_b = 0
    for W, b in weights:
        total_W += W
        total_b += b
    aggregated_W = total_W / len(weights)
    aggregated_b = total_b / len(weights)
    return (aggregated_W, aggregated_b)

def send_weights(target_host, port, weights):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((target_host, port))
    serialized_weights = pickle.dumps(weights)
    sock.sendall(serialized_weights)
    new_weight = sock.recv(10240)
    new_weight = pickle.loads(new_weight)
    print("client_weight:{}".format(new_weight))
    # sock.close()
    return new_weight