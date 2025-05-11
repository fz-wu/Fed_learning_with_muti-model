import socket
import pickle
import numpy as np
from fed_lgr.model import LogisticRegressionModel
from fed_lgr.client import Host
from fed_lgr.heart_disease_dataset import get_data, get_labels
from utils.options import args_parser  # 复用 fed_lr 的 options.py
import os
import sys

args = args_parser()

BATCH_SIZE = 10

def generate_batch_ids(N, batch_size):
    return np.random.choice(N, batch_size, replace=False)

def lgr_train():
    # Load vertically partitioned features
    x1, x2, x3 = get_data()
    y = get_labels().values.reshape(-1, 1)
    N = x1.shape[0]
    round_num = args.round
    # Connect to server and receive assigned client index
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.port))
    client_id = pickle.loads(client_socket.recv(1024))
    print(f"[Client] Assigned client index: {client_id}")
    # Initialize local client model
    if client_id == 1:
        local_data = x1
    elif client_id == 2:
        local_data = x2
    elif client_id == 3:
        local_data = x3
    else:
        raise ValueError("Invalid client_id received from server")
    host = Host(0.0001, LogisticRegressionModel, data=local_data)

    for comm_round in range(round_num):
        print(f"[Client {client_id}] Round {comm_round + 1}/{round_num}")
        ids = generate_batch_ids(N, BATCH_SIZE)

        # Forward pass and send z
        host.forward(ids)
        z_send = pickle.dumps((ids, host.send()))
        client_socket.sendall(z_send)

        # Receive diff from server
        raw_diff = client_socket.recv(102400)
        diff = pickle.loads(raw_diff)
        host.receive(diff)

        # Local gradient update
        host.compute_gradient()
        host.update_model()

    client_socket.close()

if __name__ == "__main__":
    args = args_parser()
    if args.role == 'client':
        lgr_train()
    else:
        print("Invalid role. Please specify '--role client'")
