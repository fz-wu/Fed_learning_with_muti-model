import torch
import torch.nn as nn
import torch.optim as optim
import socket
import pickle
from utils.options import args_parser
args = args_parser()
from datetime import datetime
import hashlib
from utils.datasets import save_model_weights
from utils.transmit import recvall

# CNN架构: 2层卷积
class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjust for CIFAR-10 (32x32 input)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # Pooling layer
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # Pooling layer
        x = x.view(x.size(0), -1)  # Flatten the tensor for FC layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 模型损失函数、模型训练、模型测试设置
class CNNModel():
    
    def __init__(self, learning_rate=0.001, iterations=10, num_classes=10):
        # Use CIFAR-10 dataset for data loading
        self.model = CNN(input_channels=3, num_classes=10)  # CIFAR-10 has 3 channels, 10 classes
        self.learning_rate = learning_rate
        self.epochs = iterations
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, train_loader):
        cost_list = [0] * self.epochs
        for epoch in range(self.epochs):
            running_loss = 0.0
            for data, target in train_loader:
                data, target = data.to('cuda'), target.to('cuda')

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(data)
                loss = self.loss(outputs, target)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            cost_list[epoch] = running_loss / len(train_loader)
        
        return self.model.state_dict()  # Return the updated weights
    
    def evaluate(self, test_loader):
        self.model.eval()
        self.model.to('cuda')
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to('cuda'), target.to('cuda')
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100.0 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

# 模型训练 (需要调节-学习率、epochs、标签数、GPU/CPU) (完成本地训练后 本地模型进行保存)
def cnn_train(train_loader,test_loader):
    args = args_parser()
    model = CNNModel(learning_rate=args.lr, iterations=args.epochs, num_classes=args.label_num)
    model.model.to(args.device)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.port))

    for round in range(args.round):
        print(f"\nRound {round + 1}")

        # 本地训练
        new_theta = model.fit(train_loader)
        model.evaluate(test_loader)

        # 获取训练样本数量
        local_sample_num = len(train_loader.dataset)

        # 发送 (权重, 样本数量)
        payload = {
            'weights': new_theta,
            'num_samples': local_sample_num
        }

        serialized_payload = pickle.dumps(payload)
        data_length = len(serialized_payload)

        client_socket.sendall(data_length.to_bytes(4, byteorder='big'))
        client_socket.sendall(serialized_payload)
        print(f"Sent weights and sample count ({local_sample_num}) to server.")

        try:
            length_data = recvall(client_socket, 4)
            total_length = int.from_bytes(length_data, byteorder='big')
            serialized_weights = recvall(client_socket, total_length)
            theta = pickle.loads(serialized_weights)
            model.model.load_state_dict(theta)
            print("Updated model from server.")
        except Exception as e:
            print(f"Error receiving updated weights: {e}")

    save_model_weights(model.model.state_dict())
    client_socket.close()
