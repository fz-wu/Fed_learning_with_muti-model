# muti-model Federated Learning with muti-machine communication

This project is a multi-client federated learning application that supports linear regression, logistic regression, k-means, and CNN.

## 部署说明

`curl -LsSf https://astral.sh/uv/install.sh | sh`
1. 安装uv后运行： uv init
2. uv sync
3. gunicorn fl_backend.wsgi -b 0.0.0.0:8000

## 命令行参数说明

| 参数名           | 类型   | 默认值      | 适用角色      | 说明                                   |
| ---------------- | ------ | ----------- | ------------- | -------------------------------------- |
| `--role`         | str    | client          | 所有          | 运行角色，`server` 或 `client`         |
| `--server_ip`    | str    | 127.0.0.1   | client        | 服务器 IP 地址（仅客户端需要）         |
| `--port`         | int    | 10000       | client        | 服务器端口号（仅客户端需要）           |
| `--client_num`   | int    | 3           | server        | 客户端数量（仅服务器需要）             |
| `--dataset`      | str    | 无          | client        | 数据集名称（如 `traindata.csv`，仅客户端需要） |
| `--model`        | str    | lr          | 所有          | 模型名称（lr lgr kmeans svm cnn）          |
| `--round`        | int    | 10          | 所有          | 训练轮数                               |
| `--label_num`    | int    | 2           | 所有          | 标签数量（分类任务用）                 |
| `--lr`           | float  | 0.01        | 所有          | 学习率（仅梯度下降法相关模型用）       |
| `--batch_size`   | int    | 32          | client        | 批处理大小            |
| `--epochs`       | int    | 10          | client        | 本地训练轮数             |
| `--device`       | str    | cuda        | client        | 训练设备，选择 `'cpu'` 或 `'cuda'`

---

### 示例用法

**服务器启动：**

`python main.py --role server --client_num 3 --model lr --round 10 --label_num 2 --lr 0.01`

客户端启动：

`python main.py --role client --server_ip 127.0.0.1 --port 10000 --dataset traindata.csv --model lr --round 10 --label_num 2 --lr 0.01`

参数说明补充
--role：指定运行角色，server 为服务器，client 为客户端。
--server_ip、--port：客户端需指定服务器的 IP 和端口。
--client_num：服务器需指定客户端数量。
--dataset：客户端需指定数据集文件名，需放在 datasets/ 目录下。
--model：模型名称需与 models/ 目录下的模型文件对应。
--round：训练轮数。
--label_num：分类任务的标签数量。
--lr：学习率，仅对梯度下降法相关模型有效。
--batch_size：客户端需指定批处理大小。  
--epochs：客户端需指定本地训练轮数，用于控制每轮通信前的训练次数。  
--device：客户端需指定训练设备，可选 'cpu' 或 'cuda'，默认使用 GPU（cuda）。
如需更多参数说明，请参考 utils/options.py 文件中的 args_parser() 实现。
### 模型推荐
回归问题使用 线性回归
特征复杂的二分类问题使用 SVM
特征不太复杂的多分类问题使用 逻辑回归
无监督数据使用 K-means
图像分类问题使用 CNN

### CNN 图像分类模型

本项目实现了一个用于图像分类任务的轻量级卷积神经网络（CNN），适用于自定义图像数据集。模型可用于联邦学习场景，支持 GPU 加速。

**模型架构**

- **Conv1**：输入 3 通道，输出 32 通道，卷积核 3×3，ReLU 激活  
- **MaxPool1**：2×2 最大池化  
- **Conv2**：输入 32 通道，输出 64 通道，卷积核 3×3，ReLU 激活  
- **MaxPool2**：2×2 最大池化  
- **Flatten**：展平为一维向量  
- **FC1**：输入 4096，输出 128，ReLU 激活  
- **FC2**：输入 128，输出类别数（默认 10）

**数据集要求**

- 彩色图像将被裁剪为 **32×32** 尺寸
- 所有图像按类别放入子文件夹中，文件夹名为类别标签  
- 数据目录结构示例：
```
datasets/
└── your_dataset/
    ├── class_a/
    │   ├── img1.png
    │   ├── img2.png
    ├── class_b/
    │   ├── img1.png
    │   ├── img2.png
```
**当前测试所用数据集链接** 
- [cifar-10 Python version (.tar.gz)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- [cinic-10 Python version (.tar.gz)](https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz)
- 文件解压到datasets文件夹下，并以小写cifar-10/cinic-10命名 (或确保--dataset 名称 与 数据集名称对应)

## POST请求说明

1. post方法请求 https://ip_addr/run_federated/ 运行联邦学习算法
2. get方法请求 https://ip_addr/get_models/ 获取训练模型列表 
