import numpy as np
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from server import Server
from model import LogisticRegressionModel
from client import Host
import heart_disease_dataset as hdd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,f1_score,accuracy_score
# Constants
NUM_CLIENTS = 3
BATCH_SIZE = 10
COMM_ROUNDS = 1500

def generate_batch_ids(limit, n_samples, batch_size, seen_samples):
    ids = [e for e in random.sample(range(limit), n_samples) if e not in seen_samples]
    seen_samples.update(ids[:batch_size])
    return ids[:batch_size]

def lgr_train():
    x1, x2, x3 = hdd.get_data()
    
    y = hdd.get_labels()
    N = x1.shape[0]
    num_batch = N // BATCH_SIZE
    
    server = Server(0.0001, LogisticRegressionModel, data=(x1, y))
    client1 = Host(0.0001, LogisticRegressionModel, data=x2)
    client2 = Host(0.0001, LogisticRegressionModel, data=x3)

    loss_per_epoch = []

    for r in range(COMM_ROUNDS):
        seen_samples = set()
        losses = []
        for _ in range(num_batch):
            ids = generate_batch_ids(N, N, BATCH_SIZE, seen_samples)
            server.forward(ids)
            client1.forward(ids)
            client2.forward(ids)

            server.receive(client1.send(), client2.send())
            server.compute_gradient()

            diff = server.send()
            client1.receive(diff)
            client2.receive(diff)

            client1.compute_gradient()
            client2.compute_gradient()
            server.update_model()
            client1.update_model()
            client2.update_model()

            losses.append(server.loss)
        
        epoch_loss = sum(losses) / len(losses) if losses else 0
        loss_per_epoch.append(epoch_loss)

    x1_test,x2_test,x3_test = hdd.get_testdata()
    
    predictions_local = server.predict_local(x1_test)
    print("Local Model Accuracy of server: ",accuracy_score(hdd.get_testlabels(),predictions_local))
    print("Local Model F1 Score of server: ",f1_score(hdd.get_testlabels(),predictions_local))
    print("Local Report",classification_report(hdd.get_testlabels(),predictions_local))

    # The server receives the contributions and makes the final prediction
    client1_contribution = client1.compute_contribution(x2_test)
    client2_contribution = client2.compute_contribution(x3_test)
    predictions = server.predict(x1_test, [client1_contribution, client2_contribution])
    print("Federated Model Accuracy of server: ",accuracy_score(hdd.get_testlabels(),predictions))
    print("Federated Model F1 Score of server: ",f1_score(hdd.get_testlabels(),predictions))
    print("Federated Report",classification_report(hdd.get_testlabels(),predictions))


    plot_loss(loss_per_epoch)


def plot_loss(loss_per_epoch):
    plt.plot(loss_per_epoch)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()

if __name__ == "__main__":
    lgr_train()
