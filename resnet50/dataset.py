from PIL import Image
import torch

class Dataset(object):
    def __init__(self, root, txt_path, class_to_idx, transforms=None):
        self.class_to_idx = class_to_idx
        self.root = root
        im_path = []
        label = []
        with open(txt_path, 'r') as infile:
            lines = infile.readlines()
            for line in lines:
                elems = line.split(' ')
                im_path.append(root + '/' + elems[0] + '_bbox.jpg')
                label.append(elems[2])
        infile.close()

        #ignore sub classes
        if(label=='T2b'):
            label = 'T2'
        elif(label=='T3a' or label=='T3b'):
            label = 'T3'
        self.label = label
        self.im_path = im_path
        self.transforms = transforms
    
    def __getitem__(self, index):
        im_path = self.im_path[index]
        label = self.label[index][:2]
        label = self.class_to_idx[label]
        label = torch.tensor(label, dtype=torch.long)
        img = Image.open(im_path)
        if(self.transforms is not None):
            img = self.transforms(img)
        
        return img, label
    
    def __len__(self):
        return len(self.im_path)
        
