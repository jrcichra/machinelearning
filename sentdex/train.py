#!python3
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

DIMS = 50


class Net(nn.Module):
    def __init__(self):
        super().__init__()  # just run the init of parent class (nn.Module)
        # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 conv
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        x = torch.randn(DIMS, DIMS).view(-1, 1, DIMS, DIMS)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        # 512 in, 5 out bc we're doing 5 classes
        self.fc2 = nn.Linear(512, 5)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        # .view is reshape ... this flattens X before
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)


if torch.cuda.is_available():
    # you can continue going on here, like cuda:1 cuda:2....etc.
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

X = torch.Tensor([i[0] for i in training_data]).view(-1, DIMS, DIMS)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])


print(X[0])
print(y[0])
# plt.imshow(X[1], cmap="gray")
# plt.show()

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

net = Net().to(device)
print(net)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

MODEL_NAME = f"model-{int(time.time())}"


def train(net):
    BATCH_SIZE = 100
    EPOCHS = 10
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            data = []
            # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                # print(f"{i}:{i+BATCH_SIZE}")
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, DIMS, DIMS)
                batch_y = train_y[i:i+BATCH_SIZE]
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    d = f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n"
                    data.append(d)
            f.write(''.join(data))


def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()


def fwd_pass(X, y, train=False):

    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss


def test(size=32):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(
        X.view(-1, 1, DIMS, DIMS).to(device), y.to(device))
    return val_acc, val_loss
