from utils import args_parser
from utils import create_connect

from fed_lr.main import lr_train
from fed_lgr.main import lgr_train
from fed_kmeans.main import kmeans_train    
from fed_svm.main import svm_train
from fed_cnn.main import cnn_train

def client_train():
    print(args_parser())
    args = args_parser()
    if args.model == 'lr':
        lr_train()
    elif args.model == 'lgr':
        lgr_train()
    elif args.model == 'k-means':
        kmeans_train()
    elif args.model == 'svm':
        svm_train()
    elif args.model == 'cnn':
        cnn_train()

if __name__ == "__main__":
    args = args_parser()
    if args.role == 'server':
        # Start the server
        print("Starting listening server...")
        # server()
        create_connect(5, 10000)
    elif args.role == 'client':
        # Start the client
        print("Starting client...")
        client_train()
    else:
        print("Invalid role. Please specify 'server' or 'client'.")
    # train()
