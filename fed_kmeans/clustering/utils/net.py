import socket
import threading
import pickle
import numpy as np

# 全局变量，用于存储客户端发送的参数
client_weights = []
client_sockets = []  # 用于存储客户端的 socket 对象
client_count = 0
client_lock = threading.Lock()  # 用于保护对 client_weights 和 client_count 的访问
all_clients_connected_event = threading.Event() # 用于通知主线程所有客户端都已连接
NUM_ROUNDS = 10  # 联邦学习迭代轮数

def tcplink(sock, addr, client_id, client_num):
    global client_weights, client_count

    print('Accept new connection from {}'.format(addr))
    try:
        # 1. 接收客户端发送的参数
        weight = sock.recv(10240)
        if not weight:
            print(f"Client {client_id} disconnected.")
            return  # 客户端断开连接，直接返回
        weight = pickle.loads(weight)
        print("Received weight from {}: {}".format(addr, weight))

        with client_lock:
            client_weights.append(weight)
            client_count += 1

            # 如果所有客户端都已连接，则设置事件
            if client_count == client_num:
                print("All clients connected. Proceeding to aggregation.")
                all_clients_connected_event.set()

    except Exception as e:
        print("Error in tcplink:", e)
    # finally:
    #     sock.close()
    #     print('Connection from %s:%s closed.' % addr)

def create_connect(client_num, port):
    global client_weights, client_count, all_clients_connected_event, client_sockets
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print("client number = {}".format(client_num))
    
    print('Waiting for connection...')
    #创建一个线程池
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

    # 主循环：迭代进行联邦学习
    for round in range(NUM_ROUNDS):
        print(f"Starting aggregation round {round + 1}...")
        all_clients_connected_event.wait()  # 等待所有客户端连接
        all_clients_connected_event.clear()  # 清除事件，为下一轮做准备
        
        print("All clients connected. Starting aggregation...")
        print("client_weights:{}".format(client_weights))
        # 模拟参数聚合
        aggregated_weights = aggregate_weights(client_weights)
        print("Aggregated weights:", aggregated_weights)

        # 将聚合后的权重发送给所有客户端
        with client_lock:
            client_weights_copy = client_weights[:]  # 创建 client_weights 的副本
            client_weights = []  # 清空 client_weights
            client_count = 0  # 重置 client_count

        for client_socket in client_sockets:
            try:
                # 发送聚合后的权重到客户端
                serialized_weights = pickle.dumps(aggregated_weights)
                client_socket.sendall(serialized_weights)
                print("Sent aggregated weights to client {}.".format(client_socket.getpeername()))
            except Exception as e:
                print("Error sending weights to client:", e)

        print("Weights sent to all clients. Ready for the next round.")

def aggregate_weights(weights):
    """
    模拟参数聚合的函数。
    在实际应用中，你需要根据联邦学习算法来实现参数聚合。
    """
    # 示例：简单地将所有权重相加
    # 初始化权重和偏置的累加器
    print("wait weights:{}".format(weights))
    # total_W = np.zeros_like(weights[0][0])
    # total_b = np.zeros_like(weights[0][1])
    # # 累加所有客户端的权重和偏置
    # for weight in weights:
    # total_W += weight[0][0]
    # total_b += weight[0][1]

    # # 计算平均权重和偏置
    # aggregated_W = total_W / len(weights)
    # aggregated_b = total_b / len(weights)

    aggregated_weights = (weights[0])
    return aggregated_weights

def send_weights(target_host, port, weights):
    # Create a socket connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((target_host, port))

    # Serialize the weights using pickle
    serialized_weights = pickle.dumps(weights)

    # Send the serialized weights over the socket
    sock.sendall(serialized_weights)
    new_weigth = sock.recv(10240)
    new_weigth = pickle.loads(new_weigth)
    print("client_weigth:{}".format(new_weigth))
    # Close the socket
    # sock.close()
    return new_weigth