import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from dotenv import find_dotenv, load_dotenv
import os
import sys
from utils.net import send_weights
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ML import Model
from client import Participant
from server import server
import dataset as dd
import pickle
client1_localdata=dd.client1_data()
client2_localdata=dd.client2_data()
client3_localdata=dd.client3_data()

# Local Data for Each Clients


M, N = client1_localdata[0].shape
communication_rounds = 100

def predict(X,theta):
    return np.dot(X,theta[0]) + theta[1]

def lr_train(X, Y):

        # load env 
    load_dotenv(verbose=True)
    print(os.getenv("ip_client1",default=None))
    # Create client instances

    M, N = X.shape
    theta = (np.zeros((N,1)),0)
    # Create server instance
    # S = server(initial_global_model)
    model = Model(data=(X, Y), learning_rate=0.001, iterations=10) # 读数据加载模型
    theta  = model.fit(theta)  # 训练模型
    print("train_theta:{}".format(theta[0]))
    # print(os.getenv("ip_client1",default=None),port=os.getenv("port_client1",default=None))
    new_weight = send_weights(weights=theta, target_host="127.0.0.1", port=10000) # 发送模型参数到客户端
    print("new_weight:{}".format(new_weight))
    # new_theta = pickle.loads(new_weight) # 接收客户端返回的模型参数
    # print("new weight:".format(new_theta[0]))
    # for round in range(communication_rounds):

    #     theta = model.fit(theta)  
    #     print(theta[0])
        # client1.train()

if __name__=="__main__":

    # Create client instances
    client1 = Participant(Model, data=client1_localdata) # 读数据加载模型
    client2 = Participant(Model, data=client2_localdata)
    client3 = Participant(Model, data=client3_localdata)

    initial_global_model = (np.zeros((N,1)),0) # 初始化模型
    # Create server instance
    S = server(initial_global_model)
    
    for round in range(communication_rounds):
        # print(round)
        theta = S.send_to_clients()         #In the first iteration the initial_global_model is send to all the clients from the server

        client1.receive_from_server(theta)
        client2.receive_from_server(theta)    # Clients receive global model from server and update local model
        client3.receive_from_server(theta)    # Typically this should happen parallely but it is sequential here


        client1.train()
        client2.train()     # Clients parallely train local models using local data
        client3.train()
        
        if round%10==0:

            print('client 1',client1.theta[0])    # Check local model at any iteration
            # print('client 2',client2.theta[0])
            # print('client 3',client3.theta[0])

        S.receive_from_clients(client1.send_to_server(), client2.send_to_server(), client3.send_to_server()) # Server receives updated local models for aggregation

    # pickle.dump(S.theta, open('fed_lr.pkl',"wb")) # Save the global model after training

    # X_test, y_test =dd.get_testdata()
    # y_pred = predict(X_test,client1.theta)  # Can be used to predict the testdata
    # print(r2_score(y_test,y_pred))
