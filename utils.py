import socket
import os
import argparse
import threading
import time  
import pandas as pd

def load_datasets(dataset, data_dir):
    df_train = pd.read_csv(os.path.join(data_dir, dataset))
    X = df_train.iloc[:,:-1].values
    Y = df_train.iloc[:,-1:].values
    return X, Y

def tcplink(sock, addr):
    print('Accept new connection from {}'.format(addr))
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    # sock.close()
    print('Connection from %s:%s closed.' % addr)

def create_connect(client_num, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(client_num)
    print('Waiting for connection...')
    while True:
        # 接受一个新连接:
        client_socket, client_addr = server_socket.accept()
        print("client_socket: {}, addr: {}".format(client_socket, addr))
        # 创建新线程来处理TCP连接:
        t = threading.Thread(target=tcplink, args=(client_socket, addr))
        t.start()

def send_weights(target_host, port, weights):
    # Create a socket connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((target_host, port))

    # Serialize the weights using pickle
    serialized_weights = pickle.dumps(weights)

    # Send the serialized weights over the socket
    sock.sendall(serialized_weights)

    # Close the socket
    # sock.close()




def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--role', type=str, )
    parser.add_argument('--dataset', type=str, help='name of dataset')
    parser.add_argument('--model', type=str, default='lr', help='model name')
    parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=3, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    args = parser.parse_args()
    return args