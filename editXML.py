# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:31:13 2019

@author: tians
"""

import xml.etree.ElementTree as ET

tree = ET.parse('C:\\Users\\tians\\Desktop\\TCGA-ZF-AA5N18.xml')
root = tree.getroot()

folder = root.find('folder')
print(folder.text)
folder.text = ('VOC2007')

tree.write('out.xml')