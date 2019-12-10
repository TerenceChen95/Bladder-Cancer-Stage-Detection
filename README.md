# Bladder-Cancer-Stage-Detection
## Faster-RCNN Bladder Detection
Detect bladder and subtract the bladder window (this code is updating)
## DCGAN sampler
Generate GAN samples for T0, T1 and T4 in order to balance the dataset, here's the comparison between original and generated dataset sub-classes samples count:
![](./eval/compare.png)
## Resnet50
Classify stages of bladder cancer
## Evaluation
ROC and AUC of individual class are calculated, beside recall, precision and F1 showed as report
![](./eval/ROC.jpg)
![](./eval/report.png)
