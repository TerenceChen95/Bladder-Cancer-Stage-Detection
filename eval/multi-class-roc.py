fprs = []
tprs = []
aucs= []
labels = []
with open('ROCs.txt', 'r') as infile:
    lines = infile.readlines()
    for line in lines:
        words = line.split(']')
        
        w_fpr = words[0].strip('[')
        fpr = w_fpr.split(' ')
        fpr = list(filter(None, fpr))
        
        w_tpr = words[1]
        tpr = w_tpr.split(' ')
        tpr = list(filter(None, tpr))
        tpr[0] = tpr[0][1:]

        auc = words[2]
        auc_lab = auc.split(' ')
        auc = auc_lab[1]
        label = auc_lab[2].strip('\n') 
        
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc)
        labels.append(label)
infile.close()


#transfer string list to numpy array
for fpr in fprs:
    for i,elem in enumerate(fpr, 0):
        elem = float(elem)
        fpr[i] = elem

for tpr in tprs:
    for i,elem in enumerate(tpr, 0):
        elem = float(elem)
        tpr[i] = elem
        
import numpy as np
import matplotlib.pyplot as plt

c_names = ['maroon', 'brown', 'coral', 'darkolivegreen', 'darkviolet']
plt.figure()
plt.plot([0,1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim = ([0.0, 1.0])
plt.ylim = ([0.0, 1.05])
for i, fpr in enumerate(fprs, 0):
    plt.plot(fpr, tprs[i], c=c_names[i], lw=2, alpha=0.7, label='AUC_%s=%0.3f'%(labels[i], float(aucs[i])))
    
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC for stages')
plt.legend(loc='lower right')
plt.savefig('ROC.jpg')









