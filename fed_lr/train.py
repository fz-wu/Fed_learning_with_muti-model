from json import load
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
import socket

from utils.datasets import load_datasets, get_dataset_path, save_model_weights
from utils.options import args_parser
args = args_parser()

# from client import Participant
# from server import server
# import  dataset as dd
import pickle


# Local Data for Each Clients
# M, N = client1_localdata[0].shape

class Model():
    
    def __init__( self, data, learning_rate=0.001, iterations=10) :
        self.x = data[0]
        self.y = data[1]
        # self.w = theta[0]
        self.w =  np.zeros((self.x.shape[1],1))
        self. b = 0
        self.learning_rate = learning_rate

        self.epochs = iterations
        
    def loss(self):
        cost = np.sum((((self.x.dot(self.w) + self.b) - self.y) ** 2) / (2*len(self.y)))
        return cost

    def fit(self,theta):
        self.w = theta[0]
        self.b = theta[1]
        cost_list = [0] * self.epochs
    
        for epoch in range(self.epochs):
            z = self.x.dot(self.w) + self.b
            loss = z - self.y
            
            weight_gradient = self.x.T.dot(loss) / len(self.y)
            bias_gradient = np.sum(loss) / len(self.y)
            
            self.w = self.w - self.learning_rate*weight_gradient
            self.b = self.b - self.learning_rate*bias_gradient
    
            cost = self.loss()
            cost_list[epoch] = cost
            
            # if (epoch%(self.epochs/10)==0):
            #     print("Cost is:",cost)
            
        return self.w, self.b


def predict(X,theta):
    return np.dot(X,theta[0]) + theta[1]

def lr_train():
    args = args_parser()

    datasets_path = get_dataset_path()
    X, Y = load_datasets(datasets_path)
    round = args.round
        # load env 
    # load_dotenv(verbose=True)
    # print(os.getenv("ip_client1",default=None))
    # Create client instances

    M, N = X.shape
    theta = (np.zeros((N,1)),0)
    # Create server instance
    # S = server(initial_global_model)
    model = Model(data=(X, Y), learning_rate=args.lr, iterations=10) # 读数据加载模型
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip,args.port))

    for _ in range(round):
        print("round:{}".format(_))
        # 1. train model
        # 2. send theta to server
        # 3. receive theta from server
        # 4. update theta
        # 5. repeat

        new_theta = model.fit(theta)  # 训练模型
        new_weight = pickle.dumps(new_theta)
        client_socket.sendall(new_weight)
        print("send_theta:{}".format(new_theta))

        # print("theta:{}".format(theta))

        # 1. receive theta from server
        weights = client_socket.recv(10240)
        theta = pickle.loads(weights)
        print("new_theta:{}".format(new_theta))
        # X_test, y_test =dd.get_testdata()
        # y_pred = predict(X_test, theta)  # Can be used to predict the testdata
        # print("Acc: {}".format(r2_score(y_test,y_pred)))

        # model.train()
        # print("train_theta:{}".format(theta[0]))
        
    # theta 
    test_data_path = datasets_path.replace("train", "test")
    print(test_data_path)
    X_test, y_test = load_datasets(test_data_path)
    y_pred = predict(X_test, theta)  # Can be used to predict the testdata
    print('The r2_score is {}'.format(r2_score(y_test,y_pred)))
    save_model_weights(theta)
    print("All round have finished. Save model weights success.")
    client_socket.close()
    print("close socket")
