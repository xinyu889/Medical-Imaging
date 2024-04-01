import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import time
import os
import torchvision.models as models
from tqdm import tqdm
from typing import Literal
from functools import reduce
import csv
from torchvision.models import DenseNet121_Weights,ResNet50_Weights,VGG19_Weights,GoogLeNet_Weights,AlexNet_Weights
from Model_coatnet import CoAtNet,coatnet_0

#Dataset
def load_patients(csv_path, data_dir_path):
    patients = {}
    with open(csv_path, encoding='unicode_escape') as csvFile:
        csvDictReader = csv.DictReader(csvFile)
        for row in csvDictReader:
            pid = row["Patient ID"]
            if patients.get(pid) is None:
                patients[pid] = []
            patients[pid].append(os.path.join(data_dir_path, row["File name"]))

    return [patient for patient in patients.values()]


def percent_list_slice(x, start=0., end=1.):
    return x[int(len(x) * start):int(len(x) * end)]


class CovidCT(Dataset):
    def __init__(self,
                 data_root,
                 mode: Literal["train", "valid", "test"] = "train",
                 transform=None):
        if mode == "train":
            start, end = 0.0, 0.6
        elif mode == "valid":
            start, end = 0.6, 0.8
        elif mode == "test":
            start, end = 0.8, 1.0

        normal_patients = load_patients(
            os.path.join(data_root, "meta_data_normal.csv"),
            os.path.join(data_root, "curated_data/curated_data/1NonCOVID"))
        normal_patients = percent_list_slice(normal_patients, start, end)
        normal_file_paths = reduce(lambda a, b: a + b, normal_patients)

        covid_patients = load_patients(
            os.path.join(data_root, "meta_data_covid.csv"),
            os.path.join(data_root, "curated_data/curated_data/2COVID"))
        covid_patients = percent_list_slice(covid_patients, start, end)
        covid_file_paths = reduce(lambda a, b: a + b, covid_patients)

        self.file_paths = normal_file_paths + covid_file_paths
        self.labels = [0] * len(normal_file_paths) + [1] * len(covid_file_paths)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = Image.open(self.file_paths[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]

#train
CUDA_DEVICES = 0
init_lr = 0.01

# Save model every 5 epochs
checkpoint_interval = 10
if not os.path.isdir('./Checkpoint/'):
    os.mkdir('./Checkpoint/')


# Setting learning rate operation
def adjust_lr(optimizer, epoch):
    # 1/10 learning rate every 5 epochs
    lr = init_lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    # If out of memory , adjusting the batch size smaller
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trainset = CovidCT("D:/bocheng/homework/archive/", "train", data_transform)
    train_dl = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=3)
    validset = CovidCT("D:/bocheng/homework/archive/", "valid", data_transform)
    valid_dl = DataLoader(validset, batch_size=32, shuffle=False, num_workers=3)
    classes = ['1NonCOVID', '2COVID']

    #model = models.resnet18(pretrained=True)
    #model = models.densenet121(weights= DenseNet121_Weights.IMAGENET1K_V1)
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    #model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    #model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    #model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    #model = coatnet_0()
    #model.fc = nn.Linear(in_features=512, out_features=2, bias=True)  # 如果要使用預訓練模型，記得修改最後一層輸出的class數量
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512, out_features=2, bias=True)
    )
    print(model)
    print("==========")

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    model = model.cuda(CUDA_DEVICES)

    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Training epochs
    num_epochs = 200
    criterion = nn.CrossEntropyLoss()


    # Optimizer setting
    optimizer = torch.optim.SGD(params=model.parameters(), lr=init_lr,momentum=0.9)

    # Log
    with open('TrainingAccuracy.txt', 'w') as fAcc:
        print('Accuracy\n', file=fAcc)
    with open('TrainingLoss.txt', 'w') as fLoss:
        print('Loss\n', file=fLoss)

    for epoch in range(num_epochs):
        model.train()
        localtime = time.asctime(time.localtime(time.time()))
        print('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1, num_epochs, localtime))
        print('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1, num_epochs, localtime)))

        training_loss = 0.0
        training_corrects = 0
        adjust_lr(optimizer, epoch)

        for i, (inputs, labels) in (enumerate(tqdm(train_dl))):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += float(loss.item() * inputs.size(0))
            training_corrects += torch.sum(preds == labels.data).item()

        training_loss = training_loss / len(trainset)
        training_acc = training_corrects / len(trainset)
        print('\n Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(training_loss, training_acc))

        # Check best accuracy model ( but not the best on test )
        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

        with open('TrainingAccuracy.txt', 'a') as fAcc:
            print('{:.4f} '.format(training_acc), file=fAcc)
        with open('TrainingLoss.txt', 'a') as fLoss:
            print('{:.4f} '.format(training_loss), file=fLoss)
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model, './Checkpoint/model-epoch-{:d}-train.pth'.format(epoch + 1))

        model = model.cuda(CUDA_DEVICES)
        model.eval()
        total_correct = 0
        total = 0
        class_correct = list(0. for i in enumerate(classes))
        class_total = list(0. for i in enumerate(classes))

        with torch.no_grad():
            for inputs, labels in tqdm(valid_dl):
                inputs = Variable(inputs.cuda(CUDA_DEVICES))
                labels = Variable(labels.cuda(CUDA_DEVICES))
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()

                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

            for i, c in enumerate(classes):
                if (class_total[i] == 0):
                    print('Accuracy of %5s : %8.4f %%' % (
                        c, 100 * 0))
                else:
                    print('Accuracy of %5s : %8.4f %%' % (
                        c, 100 * class_correct[i] / class_total[i]))

            # Accuracy
            print('\nAccuracy on the ALL val images: %.4f %%'
                  % (100 * total_correct / total))

    # Save best training/valid accuracy model ( not the best on test )
    model.load_state_dict(best_model_params)
    best_model_name = './Checkpoint/model-{:.2f}-best_train_acc.pth'.format(best_acc)
    torch.save(model, best_model_name)
    print("Best model name : " + best_model_name)


if __name__ == '__main__':
    train()