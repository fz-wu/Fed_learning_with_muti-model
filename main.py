from utils.options import args_parser
from utils.net import create_connect, create_connect_cnn, create_connect_svm, create_connect_lgr

from fed_lr.train import lr_train
from fed_lgr.train import lgr_train
# from fed_kmeans.train import kmeans_train
# from fed_svm.main import svm_train
from fed_svm.train import svm_train
from fed_cnn.train import cnn_train
import os 

def client_train():
    args = args_parser()
    if args.model == 'lr':
        lr_train()
    elif args.model == 'lgr':
        lgr_train()
    elif args.model == 'kmeans':
        print("kmeans")
        # kmeans_train()
        # fed_kmeans()
        # test_federated()
    elif args.model == 'svm':
        svm_train()
    elif args.model == 'cnn':
        cnn_train()

if __name__ == "__main__":
    args = args_parser()
    if args.role == 'server':
        # Start the server
        print("Starting listening server...")
        # server() 10000 is the port number
        if args.model == 'cnn':
            create_connect_cnn(args.client_num, args.port)
        elif args.model == 'svm':
            create_connect_svm(args.client_num, args.port)
        elif args.model == 'lgr':
            create_connect_lgr(args.client_num, args.port)      
        else:
            create_connect(args.client_num, args.port)

    elif args.role == 'client':
        # Start the client
        print("Starting client...")
        client_train()
    else:
        print("Invalid role. Please specify 'server' or 'client'.")
    # train()
