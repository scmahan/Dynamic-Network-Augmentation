import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms as T
from sklearn.model_selection import StratifiedShuffleSplit


def get_data(data, dataroot, train=True, seed=0):
    if data=="CIFAR10":
        return CIFAR10(dataroot, train=train)
    elif data=="ReducedCIFAR10":
        return ReducedCIFAR10(dataroot, dataseed=seed)
    elif data=="CIFAR100":
        return CIFAR100(dataroot, train=train)
    elif data=="ReducedCIFAR100":
        return ReducedCIFAR100(dataroot, dataseed=seed)
    elif data=="SVHN":
        split = train*"train" + (not train)*"test"
        return SVHN(dataroot, split=split)
    elif data=="ReducedSVHN":
        return ReducedSVHN(dataroot, dataseed=seed)
    
    
def num_class(dataset):
    return {
        'CIFAR10': 10,
        'ReducedCIFAR10': 10,
        'CIFAR100': 100,
        'ReducedCIFAR100': 100,
        'SVHN': 10,
        'ReducedSVHN': 10,
    }[dataset]
    
    
class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    
    
class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, download=True, **kwargs)
        with torch.no_grad():  
            if self.train:
                self.transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor()
                ])
                self.transform.transforms.append(CutoutDefault(16))
            else:
                self.transform = T.ToTensor()
              
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        img = self.transform(img)
        return img, int(self.targets[idx])
    
    
class ReducedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, size=4000, dataseed=0, **kwargs):
        super().__init__(*args, download=True, **kwargs)
        with torch.no_grad():  
            if self.train:
                self.transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ])
                self.transform.transforms.append(CutoutDefault(16))
            else:
                self.transform = T.ToTensor()
                
            sss = StratifiedShuffleSplit(n_splits=1, test_size=50000-size, random_state=dataseed)  
            sss = sss.split(list(range(len(self.data))), self.targets)
            train_idx, _ = next(sss)
            self.data = np.array([self.data[idx] for idx in train_idx])
            self.targets = [self.targets[idx] for idx in train_idx]
  
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        img = self.transform(img)
        return img, int(self.targets[idx])
    
    
class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, download=True, **kwargs)
        with torch.no_grad():  
            if self.train:
                self.transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor()
                ])
                self.transform.transforms.append(CutoutDefault(16))
            else:
                self.transform = T.ToTensor()
              
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        img = self.transform(img)
        return img, int(self.targets[idx])
    
    
class ReducedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, size=4000, dataseed=0, **kwargs):
        super().__init__(*args, download=True, **kwargs)
        with torch.no_grad():  
            if self.train:
                self.transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ])
                self.transform.transforms.append(CutoutDefault(16))
            else:
                self.transform = T.ToTensor()
                
            sss = StratifiedShuffleSplit(n_splits=1, test_size=50000-size, random_state=dataseed)  
            sss = sss.split(list(range(len(self.data))), self.targets)
            train_idx, _ = next(sss)
            self.data = np.array([self.data[idx] for idx in train_idx])
            self.targets = [self.targets[idx] for idx in train_idx]
  
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        img = self.transform(img)
        return img, int(self.targets[idx])
    
    
class SVHN(torchvision.datasets.SVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, download=True, **kwargs)
        with torch.no_grad():  
            self.transform = T.Compose([
                    T.ToTensor()
                ])
            if self.split == "train":
                self.transform.transforms.append(CutoutDefault(20))
      
    def __getitem__(self, idx):
        img = Image.fromarray(np.moveaxis(self.data[idx],0,2))
        img = self.transform(img)    
        return img, int(self.labels[idx])
    

class ReducedSVHN(torchvision.datasets.SVHN):
    def __init__(self, *args, size=1000, dataseed=0, **kwargs):
        super().__init__(*args, download=True, **kwargs)
        with torch.no_grad():  
            sss = StratifiedShuffleSplit(n_splits=1, test_size=73257-size, random_state=dataseed)  
            sss = sss.split(list(range(len(self.data))), self.labels)
            train_idx, _ = next(sss)
            self.data = np.array([self.data[idx] for idx in train_idx])
            self.labels = [self.labels[idx] for idx in train_idx]
            self.data = self.data.astype(np.float32)/255
            self.data = torch.tensor(self.data)
  
    def __getitem__(self, idx):
        return self.data[idx], int(self.labels[idx])
        
