import torch
import os
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义网络结构 (AlexNet)
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

# 加载Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    mnist_test = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter

# 训练和测试
def train_and_test(net, train_iter, test_iter, num_epochs, lr, device, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    train_loss_list, train_acc_list, test_acc_list = [], [], []

    for epoch in range(num_epochs):
        train_loss, train_correct, total = 0.0, 0, 0
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            train_correct += (predicted == y).sum().item()

        train_loss_list.append(train_loss / len(train_iter))
        train_acc_list.append(train_correct / total)

        test_correct, test_total = 0, 0
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                outputs = net(X)
                _, predicted = outputs.max(1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()

        test_acc_list.append(test_correct / test_total)
        print(f"Epoch {epoch+1}/{num_epochs} => "
              f"Train loss: {train_loss/len(train_iter):.4f}, "
              f"Train acc: {train_correct/total:.4f}, "
              f"Test acc: {test_correct/test_total:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.legend()
    plt.title("Loss during training")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(test_acc_list, label="Test Acc")
    plt.legend()
    plt.title("Accuracy during training")
    plt.savefig(os.path.join(save_dir, "training_plot_alexnet.png"), bbox_inches="tight")

# 设置参数并启动训练
batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_and_test(net, train_iter, test_iter, num_epochs, lr, device)
