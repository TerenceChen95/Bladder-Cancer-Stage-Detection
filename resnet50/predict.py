import torch
from PIL import Image, ImageFilter
import seaborn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable

device = torch.device('cuda:0')
#load model
model = torch.load('../BEST_checkpoint_resnet50.pth.tar')['model']
model.to(device)
model.eval()
class_to_idx = {'T0':0, 'T1':1, 'T2': 2, 'T3':3, 'T4':4}
cat_to_name = {class_to_idx[i]: i for i in list(class_to_idx.keys())}

img_pth = '/home/tianshu/bladder-cancer/dataset/bbox_images/TCGA-ZF-AA5P40_bbox.jpg'
img_i = Image.open(img_pth)
data_transforms = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Lambda(lambda image: image.filter(ImageFilter.EDGE_ENHANCE_MORE)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    
    ])

img = data_transforms(img_i)
data = img.unsqueeze_(0)
data = Variable(data)
data = data.to(device)
probs = torch.exp(model.forward(data))
top_probs, top_labs = probs.topk(3)
top_probs = top_probs.cpu().detach().numpy().tolist()[0]
top_labs = top_labs.cpu().detach().numpy().tolist()[0]

top_cat = [cat_to_name[lab] for lab in top_labs]
plt.figure(figsize=(6, 10))
plt.subplot(2,1,1)
plt.imshow(img_i)
plt.subplot(2,1,2)
seaborn.barplot(x=top_probs, y=top_cat, color=seaborn.color_palette()[0])
plt.savefig('predict_img.png')


