from sklearn.metrics import f1_score
import torch
from dataset import Dataset
from sklearn.preprocessing import label_binarize
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import numpy as np
from PIL import Image, ImageFilter
from torch.autograd import Variable

model = torch.load('../BEST_checkpoint_resnet50.pth.tar')['model']
device = torch.device('cuda:2')
model.to(device)
model.eval()
data_transforms = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Lambda(lambda image: image.filter(ImageFilter.EDGE_ENHANCE_MORE)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

class_to_idx = {'T0':0, 'T1':1, 'T2':2, 'T2b': 2, 'T3': 3, 'T3a': 3, 'T3b' : 3, 'T4':4}


n_classes  = 5
folder = '/home/tianshu/bladder-cancer/dataset/bbox_images'
test_data = '/home/tianshu/bladder-cancer/dataset/test.txt'
img_pths = []
labels = []
with open(test_data, 'r') as infile:
    lines = infile.readlines()
    for line in lines:
        words = (line.strip('\n')).split(' ')
        fn = words[0]
        label = words[2]
        img_pth = folder+'/'+fn+'_bbox.jpg'
        img_pths.append(img_pth)
        labels.append(label)
infile.close()

y_scores = []
y_trues = []
for i, img_pth in enumerate(img_pths, 0):
    img = Image.open(img_pth)
    img = data_transforms(img)
    data = img.unsqueeze_(0)
    data = Variable(data)
    data = data.to(device)
    y_true = labels[i]
    
    '''
    ground_truth = class_to_idx[y_true]
    #for recall and F1
    prob = torch.exp(model.forward(data))
    _, y_pred = prob.topk(1)
    y_pred = y_pred.cpu().detach().numpy()[0][0]
    '''
    if((y_true=='T2') or (y_true=='T2b')):
        ground_truth = 1
    else:
        ground_truth = 0

    prob = torch.exp(model.forward(data))
    #top_probs, top_labs = prob.topk(5)
    _, top1 = prob.topk(1)
    top1 = top1.cpu().detach().numpy()[0][0]
    if(top1 == 2):
        y_pred = 1
    else:
        y_pred = 0

    y_trues.append(ground_truth)
    y_scores.append(y_pred)

y_trues = np.array(y_trues)
y_scores = np.array(y_scores)

'''
from sklearn.metrics import classification_report

target_names = ['T0', 'T1', 'T2', 'T3', 'T4']
print(classification_report(y_trues, y_scores, target_names=target_names))
'''
fpr, tpr, thresholds = metrics.roc_curve(y_trues.ravel(), y_scores.ravel())
auc_score = metrics.roc_auc_score(y_trues.ravel(), y_scores.ravel())
auc = metrics.auc(fpr, tpr)
print(auc)
print(auc_score)

with open('ROCs.txt', 'a') as out:
    out.write(str(fpr) + ' ' + str(tpr) + ' ' + str(auc) + ' T2') 
out.close()



