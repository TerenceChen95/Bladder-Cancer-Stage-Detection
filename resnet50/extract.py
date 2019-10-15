from dataset import Dataset
import os
import pandas as pd
import csv

root = "/home/tianshu/bladder-cancer/dataset/CT_BLCA"
csv_path = r'/home/tianshu/bladder-cancer/dataset/bladder-cancer-label-leave-alone.xlsx'
dataste = Dataset(root=root, csv_path=csv_path)
series = os.listdir(root)

df = pd.read_excel(csv_path)
label = df['ajcc_tumor_pathologic_pt'].value_counts()
print(label)

series_l = df['bcr_patient_barcode']
l = []
for serie in series:
    index = series_l.str.find(serie)
    if index is not None:
        l.append(df.iloc[index,:3])
print(l)

with open('./extract_data.txt', 'w') as outfile:
    for line in l:
        outfile.write(str(line.values))
outfile.close()

