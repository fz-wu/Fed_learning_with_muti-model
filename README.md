# FL 

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
如需更多参数说明，请参考 utils/options.py 文件中的 args_parser() 实现。



## POST请求说明

1. post方法请求 https://ip_addr/run_federated/ 运行联邦学习算法
2. get方法请求 https://ip_addr/get_models/ 获取训练模型列表 