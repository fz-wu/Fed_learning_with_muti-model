from utils.options import args_parser
from utils.net import create_connect, create_connect_cnn, create_connect_svm

from fed_lr.train import lr_train
from fed_lgr.train import lgr_train
# from fed_kmeans.train import kmeans_train
# from fed_svm.main import svm_train
from fed_svm.train import svm_train
from fed_cnn.train import cnn_train
import os 
from utils.datasets import acquire_and_load_data, prepare_and_save_split


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
        try:
            # 数据获取
            client_id, train_loader, test_loader, locked_path, original_path = acquire_and_load_data(dataset_name=args.dataset, max_clients=args.client_num, batch_size=args.batch_size, data_dir='./data_split')
            print(f"[Client {client_id}] Data loading completed")
            # 开展本地训练
            cnn_train(train_loader=train_loader, test_loader=test_loader)
        finally:
            # 数据释放
            if os.path.exists(locked_path):
                os.rename(locked_path, original_path)



if __name__ == "__main__":
    args = args_parser()
    if args.role == 'server':
        # Start the server
        print("Starting listening server...")
        # server() 10000 is the port number
        if args.model == 'cnn':
            # 数据划分
            prepare_and_save_split(dataset_name=args.dataset, batch_size=args.batch_size, raw_data_dir=f'./datasets/{args.dataset}', save_dir='./data_split', client_num=args.client_num)
            print('Data segmentation completed')
            # 服务器运行
            create_connect_cnn(args.client_num, args.port)
        elif args.model == 'svm':
            create_connect_svm(args.client_num, args.port)      
        else:
            create_connect(args.client_num, args.port)

    elif args.role == 'client':
        # Start the client
        print("Starting client...")
        client_train()
    else:
        print("Invalid role. Please specify 'server' or 'client'.")
    # train()
