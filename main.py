from utils import args_parser
from fed_lr.broker import lr_train


def train():
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
    train()
