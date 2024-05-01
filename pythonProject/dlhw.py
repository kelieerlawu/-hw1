import gzip
import numpy as np
import matplotlib.pyplot as plt
def get_data():
    # 文件获取
    train_image = r"C:/迅雷下载/train-images-idx3-ubyte.gz"
    test_image = r"C:/迅雷下载/t10k-images-idx3-ubyte.gz"
    train_label = r"C:/迅雷下载/train-labels-idx1-ubyte.gz"
    test_label = r"C:/迅雷下载/t10k-labels-idx1-ubyte.gz" #文件路径
    paths = [train_label, train_image, test_label,test_image]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = get_data()



class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        '''
        初始化神经网络
        '''
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, X):
        '''
        进行前向传播
        '''
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        z1 = np.dot(X, W1) + b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, W2) + b2
        y = self.softmax(z2)

        return y, a1

    def backward(self, X, y_true, y_pred, a1):
        '''
        进行反向传播
        '''
        grads = {}
        m = X.shape[0]

        dz2 = y_pred - y_true
        grads['W2'] = np.dot(a1.T, dz2) / m
        grads['b2'] = np.sum(dz2, axis=0) / m

        da1 = np.dot(dz2, self.params['W2'].T)
        dz1 = da1 * (self.der_relu(a1))
        grads['W1'] = np.dot(X.T, dz1) / m
        grads['b1'] = np.sum(dz1, axis=0) / m

        return grads

    def train(self, X, y, X_val, y_val, batch_size=128, lr=0.001, num_iters=10000, reg=0.25):
        '''
        对神经网络进行训练，并记录训练过程中的损失和准确率，用于随后的可视化
        '''
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)
        loss_history = []  # 用以保存每次的损失
        val_acc_history = []  # 用以保存验证集的准确率

        for it in range(num_iters):
            # 获取训练批次
            batch_mask = np.random.choice(num_train, batch_size)
            X_batch = X[batch_mask]
            y_batch = y[batch_mask]

            # 计算梯度和损失
            y_pred, a1 = self.forward(X_batch)
            y_true = np.zeros_like(y_pred)
            y_true[np.arange(len(y_batch)), y_batch] = 1
            grad = self.backward(X_batch, y_true, y_pred, a1)
            loss = self.loss(y_true, y_pred, reg)
            print(f'iteration {it}, loss {loss}')
            loss_history.append(loss)

            # 更新参数
            for key in ('W1', 'b1', 'W2', 'b2'):
                self.params[key] -= lr * grad[key]
                if reg:  # 如果有 L2 正则化
                    if key[0] == 'W':
                        self.params[key] += -lr * reg * self.params[key]

            # 每间隔一定次数进行一次有效性验证
            if it % iterations_per_epoch == 0:
                lr *= 0.9  # 学习率退火
                acc = np.mean(self.predict(X_val) == y_val)  # 计算验证集准确率
                val_acc_history.append(acc)


        # 返回损失历史和验证集准确率历史，用于随后的可视化
        return loss_history, val_acc_history

    def predict(self, X):
        '''
        对给定输入进行预测
        '''
        y_score, _ = self.forward(X)
        y_pred = np.argmax(y_score, axis=1)

        return y_pred

    def softmax(self, x):
        '''
        Softmax函数实现
        '''
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def relu(self, x):
        '''
        ReLU函数实现
        '''
        return np.maximum(0, x)

    def der_relu(self, x):
        '''
        ReLU函数的导数实现
        '''
        return np.where(x > 0, 1, 0)

    def loss(self, y_true, y_pred, reg=0.03):
        '''
        计算交叉熵损失和 L2 正则化
        '''
        data_loss = -np.sum(y_true * np.log(y_pred + 1e-10)) / y_true.shape[0]
        reg_loss = 0.5 * reg * np.sum(self.params['W1'] ** 2) + 0.5 * reg * np.sum(self.params['W2'] ** 2)
        return data_loss + reg_loss

nn = NeuralNetwork(input_size=28*28, hidden_size=100, output_size=10)

testdata = x_test.reshape(-1, 28*28)
loss_history, val_acc_history = nn.train(X=x_train.reshape(-1, 28*28),y=y_train,X_val=testdata, y_val=y_test)

plt.figure()
plt.plot(range(len(loss_history)), loss_history)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# 画出验证集准确率曲线
plt.figure()
plt.plot(range(len(val_acc_history)), val_acc_history)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()
y_pre = nn.predict(testdata)
print(np.sum(y_pre == y_test)/10000)