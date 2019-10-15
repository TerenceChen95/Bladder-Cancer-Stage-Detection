from PIL import Image

class Dataset(object):
    def __init__(self, root, txt_path, transforms=None):
        
        self.root = root
        im_path = []
        label = []
        with open(txt_path, 'r') as infile:
            lines = infile.readlines()
            for line in lines:
                elems = line.split(' ')
                im_path.append(root + elems[0] + '.jpg')
                label.append(elems[2])
        infile.close()
        self.label = label
        self.im_path = im_path
        self.transforms = transforms
    
    def __getitem__(self, index):
        im_path = self.im_path[index]
        label = self.labels[index]
        img = Image.open(im_path)
        
        if(self.transforms is not None):
            img = self.transforms(img)
        return img, label
    
    def __len__(self):
        return len(self.img_path)
        
