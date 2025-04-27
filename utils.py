import socket
import os

def create_connect(host, port):
    pass

def send_weights(host, port, weights):
    # Create a socket connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    # Serialize the weights using pickle
    serialized_weights = pickle.dumps(weights)

    # Send the serialized weights over the socket
    sock.sendall(serialized_weights)

    # Close the socket
    sock.close()


def receive_weights(host, port, weights):
    # Create a socket connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(1)

    # Wait for a connection
    conn, addr = sock.accept()

    # Receive the serialized weights
    serialized_weights = b""
    while True:
        data = conn.recv(4096)
        if not data:
            break
        serialized_weights += data

    # Deserialize the weights using pickle
    weights = pickle.loads(serialized_weights)

    # Close the connection and socket
    conn.close()
    sock.close()

    return weights