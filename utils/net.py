import socket
import os
import argparse
import threading
import time  
import pickle
import numpy as np

def tcplink(sock, addr):
    print('Accept new connection from {}'.format(addr))
    weight = sock.recv(10240)
    weight = pickle.loads(weight)
    print("weight:{}".format(weight))
    #     M, N = weight.shape
    new_weight = (weight[0] + 1, weight[1] + 1)
    print("server_new_weight:{}".format(weight))
    sock.sendall(pickle.dumps(new_weight))
    #     time.sleep(1)
    #     if not data or data.decode('utf-8') == 'exit':
    #         break
    #     sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    # sock.close()
    # print('Connection from %s:%s closed.' % addr)

def create_connect(client_num, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print('Waiting for connection...')
    while True:
        # 接受一个新连接:
        client_socket, client_addr = server_socket.accept()

        # 创建新线程来处理TCP连接:
        t = threading.Thread(target=tcplink, args=(client_socket, client_addr))
        t.start()

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

def server_listen():
        create_connect(5, 10000)
