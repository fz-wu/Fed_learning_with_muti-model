import socket
import threading
import pickle
import numpy as np

# 全局变量，用于存储客户端发送的参数
client_weights = []
client_count = 0
client_lock = threading.Lock()  # 用于保护对 client_weights 和 client_count 的访问
all_clients_connected_event = threading.Event() # 用于通知主线程所有客户端都已连接

def tcplink(sock, addr, expected_client_num):
    global client_weights, client_count

    print('Accept new connection from {}'.format(addr))
    try:
        weight = sock.recv(10240)
        weight = pickle.loads(weight)
        print("Received weight from {}: {}".format(addr, weight))

        with client_lock:
            client_weights.append(weight)
            client_count += 1

            if client_count == expected_client_num:
                print("All clients connected!")
                all_clients_connected_event.set() # 通知主线程所有客户端都已连接

        # 等待主线程聚合参数并发送新权重
        all_clients_connected_event.wait()
        all_clients_connected_event.clear() # 重置事件

        # 接收主线程发送的新权重
        new_weight = sock.recv(10240)
        new_weight = pickle.loads(new_weight)
        print("Received new weight from server: {}".format(new_weight))

        # 发送新权重给客户端
        sock.sendall(pickle.dumps(new_weight))

    except Exception as e:
        print("Error in tcplink:", e)
    finally:
        sock.close()
        print('Connection from %s:%s closed.' % addr)

def create_connect(client_num, port):
    global client_weights, client_count
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print(client_num)
    print('Waiting for connection...')

    threads = []
    for _ in range(client_num):
        client_socket, client_addr = server_socket.accept()
        print("client_addr: {}".format(client_addr))
        t = threading.Thread(target=tcplink, args=(client_socket, client_addr, client_num))
        threads.append(t)
        t.start()
        print("Thread started for client: {}".format(client_addr))

    # 等待所有客户端连接
    for t in threads:
        t.join()

    print("All clients connected. Starting aggregation...")

    # 模拟参数聚合
    aggregated_weights = aggregate_weights(client_weights)
    print("Aggregated weights:", aggregated_weights)

    # 将聚合后的权重发送给所有客户端
    for t in threads:
        client_socket = t.args[0]
        client_socket.sendall(pickle.dumps(aggregated_weights))

    # 重置客户端参数和计数器，准备下一轮
    with client_lock:
        client_weights = []
        client_count = 0

    print("Weights sent to all clients. Ready for the next round.")

def aggregate_weights(weights):
    """
    模拟参数聚合的函数。
    在实际应用中，你需要根据联邦学习算法来实现参数聚合。
    """
    # 示例：简单地将所有权重相加
    aggregated_weights = np.mean(weights, axis=0)
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