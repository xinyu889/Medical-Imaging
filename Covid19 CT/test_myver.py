import torch
from utils import parse_args
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from dataset import IMAGE_Dataset
import numpy as np


from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score


CUDA_DEVICES = 0
num_class = 2
DATASET_ROOT = './test'
PATH_TO_WEIGHTS = './model_coatnet_20epochs_batch8_resize224_step5_angle5_nonDA_check_1.pth' # Your model name


def test():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    print(DATASET_ROOT)
    print(PATH_TO_WEIGHTS)
    test_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
    print(test_set)
    data_loader = DataLoader(
        dataset=test_set, batch_size=4, shuffle=True, num_workers=1)
    # print(data_loader)
    print(Path(DATASET_ROOT).glob('*'))
    classes = [_dir.name for _dir in Path(DATASET_ROOT).glob('*')]
    print(classes)
    classes.sort()
    classes.sort(key = len)

    # Load model
    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()
    

    total_correct = 0
    total_error = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_error = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))
    score_list = []     # 存储预测得分
    label_list = []     # 存储真实标签

    with torch.no_grad():
        
        for inputs, labels in data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_error += (predicted != labels).sum().item()

            score_tmp = outputs  # (batchsize, nclass)
 
            score_list.extend(score_tmp.detach().cpu().numpy())
            label_list.extend(labels.cpu().numpy())




            # print(total_correct)
            # print(total_error)
            c = (predicted == labels).squeeze()
            c_error = (predicted != labels).squeeze()
            # print(c_error)
            # batch size
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_error[label] += c_error[i].item()
                class_total[label] += 1
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
 
    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])
 
    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
 
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
 
    # 绘制所有类别平均的roc曲线
    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
 
    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)
 
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('ResNet50_batch_8_20Epoch.jpg')
    plt.show()

    for i, c in enumerate(classes):
        print('Accuracy of %5s : %8.4f %%' % (
        c, 100 * class_correct[i] / class_total[i]))
        
        # print('Error of %5s : %8.4f %%' % (
        # c, 100 * class_error[i] / class_total[i]))

        print('true of %5s : %d' % (
        c, class_correct[i]))
        print('false of %5s : %d' % (
        c, class_error[i]))


    # Accuracy
    print('\nAccuracy on the ALL test images: %.4f %%'
      % (100 * total_correct / total))



if __name__ == '__main__':
    test()



