from xml.dom import minidom
import os

root = '/home/tianshu/bladder-cancer/dataset'
folder = os.path.join(root, 'train_data_xml')
xml_list = os.listdir(folder)
outfile = os.path.join(folder, 'label.csv')

with open(outfile, 'w') as out:
    for xml in xml_list:
        doc = minidom.parse(os.path.join(folder, xml))
        xmin = doc.getElementByTagName('xmin')
        xmax = doc.getElementByTagName('xmax')
        ymin = doc.getElementByTagName('ymin')
        ymax = doc.getElementByTagName('ymax')
        out.writelines(doc, xmin, xmax, ymin, ymax)
out.close()