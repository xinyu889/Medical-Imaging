from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from Model_coatnet import coatnet_2
import matplotlib.pyplot as plt
import time
import os
import copy



data_transforms = {
    # 訓練資料集採用資料增強與標準化轉換
    'train': transforms.Compose([
        transforms.Resize((224,224)), # 隨機剪裁並縮放
        transforms.ToTensor(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(hue=0.5),
        # transforms.RandomHorizontalFlip(1),
        # transforms.RandomVerticalFlip(1),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 標準化
    ]),
    # 驗證資料集僅採用資料標準化轉換
    'val': transforms.Compose([
        transforms.Resize((224,224)),  # 縮放
        transforms.ToTensor(),
        # transforms.RandomRotation(5),
        # transforms.ColorJitter(hue=0.5),
        # transforms.RandomHorizontalFlip(1),
        # transforms.RandomVerticalFlip(1),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 標準化
    ]),
}


# 資料路徑
data_dir = './IMG'

# 建立 Dataset
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

# 建立 DataLoader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
               shuffle=True, num_workers=4)
               for x in ['train', 'val']}



dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# print(dataset_sizes)


class_names = image_datasets['train'].classes
# print(class_names)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time() # 記錄開始時間

    # 記錄最佳模型
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 訓練模型主迴圈
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # 對於每個 epoch，分別進行訓練模型與驗證模型
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 將模型設定為訓練模式
            else:
                model.eval()   # 將模型設定為驗證模式

            running_loss = 0.0
            running_corrects = 0

            # 以 DataLoader 載入 batch 資料
            for inputs, labels in dataloaders[phase]:
                # 將資料放置於 GPU 或 CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 重設參數梯度（gradient）
                optimizer.zero_grad()

                # 只在訓練模式計算參數梯度
                with torch.set_grad_enabled(phase == 'train'):
                    # 正向傳播（forward）
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()  # 反向傳播（backward）
                        optimizer.step() # 更新參數

                # 計算統計值
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                # 更新 scheduler
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 記錄最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # 計算耗費時間
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # 輸出最佳準確度
    print('Best val Acc: {:4f}'.format(best_acc))

    # 載入最佳模型參數
    model.load_state_dict(best_model_wts)
    return model



    # 載入 ResNet18 預訓練模型






if __name__ == '__main__':
    model_ft = models.resnet50(pretrained=True)
    # model_ft = coatnet_2()
# 取得 ResNet18 最後一層的輸入特徵數量
    num_ftrs = model_ft.fc.in_features

# 將 ResNet18 的最後一層改為只有兩個輸出線性層
# 更一般化的寫法為 nn.Linear(num_ftrs, len(class_names))
    # model_ft.fc = nn.Linear(512, 2)
    model_ft.fc = nn.Linear(num_ftrs, 2)

# 將模型放置於 GPU 或 CPU
    model_ft = model_ft.to(device)

# 使用 cross entropy loss
    criterion = nn.CrossEntropyLoss()

# 學習優化器
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

# 每 7 個 epochs 將 learning rate 降為原本的 0.1 倍
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[30,80], gamma=0.1)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=20)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
    best_model_name = 'model_resnet50_20epochs_batch8_resize224_step5_Final.pth'
    torch.save(model_ft, best_model_name)
    torch.save(model_ft, "this_is_my_model.pth")
    print("Best model name : " + best_model_name)

    # 73.52/
    # 71 74 70 77 
#  74.50/75.49/71.89