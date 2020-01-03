# Bladder-Cancer-Stage-Detection
## Faster-RCNN Bladder Detection
Thanks to Jianwei Yang and Jiasen Lu, I render their pytorch version of faster-RCNN for bladder detection and got excellent results.

@article{jjfaster2rcnn,
    Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
    Title = {A Faster Pytorch Implementation of Faster R-CNN},
    Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
    Year = {2017}
}

@inproceedings{renNIPS15fasterrcnn,
    Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
    Title = {Faster {R-CNN}: Towards Real-Time Object Detection
             with Region Proposal Networks},
    Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
    Year = {2015}
}
## Dataset
CT/MRI images are collected from TCIA-BLCA (The Cancer Imaging Archive), you can download original data from:
https://wiki.cancerimagingarchive.net/display/Public/TCGA-BLCA

These data include:
- CT/MRI Series
- CSV files regarding patient status and staging info where patientID is an important key I use for naming preprocessing images. 

Besides, I upload detect-and-crop bladder images and the GAN label here:
BaiduNetdisk Link：

https://pan.baidu.com/s/1zOp-11bdr3Tbl3W5yjAiUw 

Captcha Code ：jz7h

## Usage
1. You can try download the cropped image first, and try to run code from **dcgan** folder for T0/T1/T4 images, except that u need small changes to paths variables. 
2. After you finish all the augmentation works, try training by running:

- python main.py

Also, some changes has to be made like path variable and cuda device etc..

## DCGAN sampler
First, I choose useful slices from an CT serie where bladder is presented. But then I found some class dataset are insufficient. There are only 8 sample from T0! So I generated GAN samples for T0, T1 and T4, and merge sub-classes for T2 and T3 in order to balance the dataset, here's the comparison between original and generated dataset sub-classes samples count:

![](./eval/compare.png)

## Resnet50
Classify stages of bladder cancer, here I just use fine-tunning + pretrained model for classification.

## Evaluation
ROC and AUC of individual class are calculated, beside recall, precision and F1 showed as report
![](./eval/ROC.jpg)
![](./eval/report.png)
