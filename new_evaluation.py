from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
from PIL import Image


from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelBinarizer
import glob



PATH = sys.argv[1]


model = models.vgg13_bn(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs,2)


model.load_state_dict(torch.load(PATH))
model = model.cuda()
model.eval()

y_true = np.array([])
y_pred = np.array([])
y_oput = np.array([])

fldr = glob.glob('./path/to/test/No/*.JPG')
for file in fldr:
    #print(file)
    input = Image.open(file)
    input_resize = transforms.Resize((224,224))(input)
    input_tran = transforms.ToTensor()(input_resize)
    input_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_tran)

    input_final = input_norm.reshape(1,3,224,224)

    outputs = model(input_final.cuda())
    oput = torch.sigmoid(outputs)
    _, preds = torch.max(outputs, 1)
    #print(outputs)
    #print(preds)
    y_true = np.append(y_true,0)
    y_pred = np.append(y_pred,preds.cpu().numpy())
    y_oput = np.append(y_oput,oput.cpu().detach().numpy())


fldr = glob.glob('./path/to/test/Yes/*.JPG')
for file in fldr:
    input = Image.open(file)
    input_resize = transforms.Resize((224,224))(input)
    input_tran = transforms.ToTensor()(input_resize)
    input_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_tran)

    input_final = input_norm.reshape(1,3,224,224)

    outputs = model(input_final.cuda())
    oput = torch.sigmoid(outputs)
    _, preds = torch.max(outputs, 1)
    #print(outputs)
    #print(preds)
    y_true = np.append(y_true,1)
    y_pred = np.append(y_pred,preds.cpu().numpy())
    y_oput = np.append(y_oput,oput.cpu().detach().numpy())



print(y_true)
print(y_pred)
print(confusion_matrix(y_true, y_pred))
print('accuracy is:')
print(accuracy_score(y_true,y_pred))
print('precision is micro:')
print(precision_score(y_true,y_pred, average='micro'))
print('recall is micro:')
print(recall_score(y_true,y_pred, average='micro'))
print('F1 score is micro:')
print(f1_score(y_true,y_pred, average='micro'))
print('ROC AUC ovo is:')
print(roc_auc_score(y_true,y_pred))

