from utils.options import args_parser
from utils.net2 import create_connect, create_lgr_connect

from fed_lr.train import lr_train
# from fed_svm.train import svm_train
# from fed_lgr.main import lgr_train
from fed_kmeans.kmeans_train import kmeans_train
from fed_kmeans.clustering.kmeans_python import  fed_kmeans, test_federated
from fed_lgr.train import lgr_train
from fed_svm.main import svm_train
from fed_cnn.main import cnn_train


import os 
datasets = os.path.join(os.path.dirname('fed_lr'), 'traindata.csv')
print(datasets)
def client_train():
    print(args_parser())
    args = args_parser()
    if args.model == 'lr':
        lr_train()
    elif args.model == 'lgr':
        lgr_train()
    elif args.model == 'kmeans':
        print("kmeans")
        # kmeans_train()
        fed_kmeans()
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
        if args.model == 'lr':
            create_connect(args.client_num, args.port)
        elif args.model == 'lgr':
            create_lgr_connect(args.client_num, args.port, args.round)

    elif args.role == 'client':
        # Start the client
        print("Starting client...")
        client_train()
    else:
        print("Invalid role. Please specify 'server' or 'client'.")
    # train()
