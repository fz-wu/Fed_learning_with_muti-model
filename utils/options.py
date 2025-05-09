import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--role', type=str, )
    parser.add_argument('--server_ip', type=str, default='127.0.0.1', help='server ip') # only for client
    parser.add_argument('--server_port', type=int, default=10000, help='server port') # only for client
    parser.add_argument('--dataset', type=str, help='name of dataset') # only for client    
    parser.add_argument('--model', type=str, default='lr', help='model name')
    parser.add_argument('--lable_num', type=int, default=3, help='number of labels')
    parser.add_argument('--round', type=int, default=5, help="rounds of training")
    parser.add_argument('--client_num', type=int, default=3, help="number of users: K") # only for server
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    args = parser.parse_args()
    return args