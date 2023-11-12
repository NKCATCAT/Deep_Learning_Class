import torch
import os
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


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


def train_test_save_fig(net, train_iter, test_iter, num_epochs, lr, device, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

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
    plt.savefig(os.path.join(save_dir, "training_plot.png"), bbox_inches="tight")


# Network definition
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))

lr, num_epochs, batch_size = 0.05, 50, 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_test_save_fig(net, train_iter, test_iter, num_epochs, lr, device)
