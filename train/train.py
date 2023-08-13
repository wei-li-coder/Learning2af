import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import MobileNetV2
import torchvision.models.mobilenet
from load_data import MyDataset
import torchvision
import torch.nn.functional as F
import csv
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor()
                                 ]),
    "test": transforms.Compose([transforms.ToTensor()
                               ])}

# rootpath for dataset
root = '/data/wl/autofocus/learn2focus/dataset/'
train_dataset=MyDataset(txt=root+'train_set.txt', transform=data_transform["train"])

train_num = len(train_dataset)
print('num_of_trainData:', train_num)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=8)

test_dataset=MyDataset(txt=root+'test_set.txt', transform=data_transform["test"])
test_num = len(test_dataset)
print('num_of_testData:', test_num)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=8)

net = MobileNetV2(num_classes=49)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.999))

seed = random.randint(1,10000)
f = open('csv/progress_{}.csv'.format(seed), 'a')
csv_writer = csv.writer(f)
csv_writer.writerow(['train_loss', 'train_accuracy_0', 'train_accuracy_1', 'train_accuracy_2', 'train_accuracy_4', 'test_loss', 'test_accuracy_0', 'test_accuracy_1', 'test_accuracy_2', 'test_accuracy_4'])
f.close()

best_acc = 0.0
for epoch in range(300):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        num_classes = 49
        expand_labels = torch.tile(torch.unsqueeze(labels, 1), [1, num_classes])
        encoded_vector = torch.tile(torch.unsqueeze(torch.tensor(range(num_classes)), 0), [logits.shape[0], 1])
        criterion = -(encoded_vector - expand_labels) * (encoded_vector - expand_labels)
        criterion = criterion.type(torch.float32) / 1
        gt = F.softmax(criterion, dim=1)
        loss = torch.nn.CrossEntropyLoss()(logits, gt.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print('\nTotal step is:', step)

    # test
    net.eval()
    train_acc_0 = 0.0  # accumulate accurate number / epoch
    train_acc_1 = 0.0
    train_acc_2 = 0.0
    train_acc_4 = 0.0
    acc_0 = 0.0  # accumulate accurate number / epoch
    acc_1 = 0.0
    acc_2 = 0.0
    acc_4 = 0.0
    with torch.no_grad():
        for train_data in train_loader:
            train_images, train_labels = train_data
            outputs = net(train_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            # calculate prediction accuracy in training set
            train_acc_0 += (predict_y == train_labels.to(device)).sum().item()
            train_acc_1 += (abs(predict_y - train_labels.to(device))<=1).sum().item()
            train_acc_2 += (abs(predict_y - train_labels.to(device))<=2).sum().item()
            train_acc_4 += (abs(predict_y - train_labels.to(device))<=4).sum().item()
        train_accurate_0 = train_acc_0 / train_num
        train_accurate_1 = train_acc_1 / train_num
        train_accurate_2 = train_acc_2 / train_num
        train_accurate_4 = train_acc_4 / train_num
        print('[epoch %d] train_loss: %.3f  train_accuracy_0: %.3f   train_accuracy_1: %.3f  train_accuracy_2: %.3f  train_accuracy-4: %.3f' %
              (epoch + 1, running_loss / step, train_accurate_0, train_accurate_1, train_accurate_2, train_accurate_4))
        test_loss = 0.0
        for test_data in test_loader:
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            num_classes = 49
            expand_labels = torch.tile(torch.unsqueeze(test_labels, 1), [1, num_classes])
            encoded_vector = torch.tile(torch.unsqueeze(torch.tensor(range(num_classes)), 0), [outputs.shape[0], 1])
            criterion = -(encoded_vector - expand_labels) * (encoded_vector - expand_labels)
            criterion = criterion.type(torch.float32) / 1
            gt = F.softmax(criterion, dim=1)
            test_loss = torch.nn.CrossEntropyLoss()(outputs, gt.to(device))
            test_loss += loss.detach().cpu().numpy().item()
            # # calculate prediction accuracy in testing set
            predict_y = torch.max(outputs, dim=1)[1]
            acc_0 += (predict_y == test_labels.to(device)).sum().item()
            acc_1 += (abs(predict_y - test_labels.to(device))<=1).sum().item()
            acc_2 += (abs(predict_y - test_labels.to(device))<=2).sum().item()
            acc_4 += (abs(predict_y - test_labels.to(device))<=4).sum().item()
        test_accurate_0 = acc_0 / test_num
        test_accurate_1 = acc_1 / test_num
        test_accurate_2 = acc_2 / test_num
        test_accurate_4 = acc_4 / test_num
        print('[epoch %d] test_loss: %.3f  test_accuracy_0: %.3f   test_accuracy_1: %.3f  test_accuracy_2: %.3f  test_accuracy-4: %.3f' %
              (epoch + 1, test_loss, test_accurate_0, test_accurate_1, test_accurate_2, test_accurate_4))
        f = open('csv/progress_{}.csv'.format(seed), 'a')
        csv_writer = csv.writer(f)
        csv_writer.writerow([running_loss / step, train_accurate_0, train_accurate_1, train_accurate_2, train_accurate_4, test_loss, test_accurate_0, test_accurate_1, test_accurate_2, test_accurate_4])
        f.close()

print('Finished Training')
