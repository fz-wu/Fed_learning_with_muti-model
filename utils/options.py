import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--role', type=str, default='client', help='role of this process, client or server')

    parser.add_argument('--server_ip', type=str, default='127.0.0.1', help='server ip') # only for client
    parser.add_argument('--port', type=int, default=10000, help='server port') # only for client
    parser.add_argument('--client_num', type=int, default=3, help="number of users: K") # only for server

    parser.add_argument('--dataset', type=str, help='name of dataset') # only for client    
    parser.add_argument('--model', type=str, default='lr', help='model name')
    parser.add_argument('--round', type=int, default=10, help="rounds of training")

    parser.add_argument('--label_num', type=int, default=2, help='number of labels') # only for classification
    # parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate") # only for gardient descent
    # parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    
    # cnn 新加
    parser.add_argument('--client_id', type=int, default=1, help="client_id")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size cifar10")
    parser.add_argument('--epochs', type=int, default=10, help="local epochs cifar10")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the training on (cpu or cuda)')
    
    args = parser.parse_args()
    return args