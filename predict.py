#!/usr/bin/env python
# coding: utf-8

# In[26]:


import sys
import torch.utils.data
import torch
from glob import glob
from  torchvision import transforms,datasets
from PIL import Image
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import cv2
from torch.autograd import Variable
cwd = os.getcwd()
parser = argparse.ArgumentParser('pix4d')

## general settings.
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--batch', type=int, default=1)
config, _ = parser.parse_known_args()


# In[27]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3 ,16, 3, padding=1)
        self.conv2 = nn.Conv2d(16,32, 3, padding=1)
        self.conv3 = nn.Conv2d(32,16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 2, 5, padding=2)
        self.max = nn.MaxPool2d(2, stride=2)
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu  = nn.ReLU()
        self.softmax = nn.Softmax2d()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max(x)
        x = self.relu(self.conv3(x))
        x = self.ups(x)
        x = self.softmax(self.conv4(x))
        return x


# In[28]:


class Pix4dDataloader(torch.utils.data.Dataset):

    def __init__(self,data_dir):
        self.data_dir=data_dir
        self.image_names = glob(data_dir + "/img/*") 
        self.gt_names = glob(data_dir + "/gt/*")
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
                                           ])

    def image_loader(self,image,gt,transform):
        
        image = transform(Image.open(image).convert('RGB').resize((256, 204), Image.BILINEAR))
        gt = transform(Image.open(gt).convert('LA').resize((256, 204), Image.NEAREST)).type(torch.LongTensor)
        c,h,w = gt.shape
        label = torch.zeros((2,h,w))
        for i in range(h):
            for j in range(w):
                label[gt[0][i][j]][i][j] = 1
        return image,label
    
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        images = '{}'.format(self.image_names[idx])
        gts = '{}'.format(self.gt_names[idx])
        images,gts = self.image_loader(images,gts,transform=self.transforms)
        return [images,gts]


# In[29]:


"""
Predict
"""
def predict(net,test_loader):
        for i, (imgs,labels) in enumerate(test_loader):
            
            if config.use_gpu:
                imgs = Variable(imgs.cuda()).float()
                labels = Variable(labels.cuda()).float()

            else:
                imgs = Variable(imgs).float()
                labels = Variable(labels).float()


            out = net(imgs)

            image = out.max(1)[1]
            if config.use_gpu:
                image = image.cpu().data.numpy().squeeze()
            else:
                image = image.data.numpy().squeeze()
            
            image = cv2.resize(image, dsize=(2022,1608), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('result/predictions.jpg',255*image)
                
        return 0



if __name__ == '__main__':
    path_dic={
        "train":'images',
        "test": 'images'
    }

    """
    Data loading
    """
    
    test = Pix4dDataloader(path_dic["test"])
    test_loader = torch.utils.data.DataLoader(test, batch_size=config.batch, num_workers=1)
    
    if config.use_gpu:
        net=Net().cuda()
    else:
        net=Net()
    print("loading the model")
    epoch = 4995
    net.load_state_dict(torch.load("trained_model/Net_"+str(epoch)+".pkl"))
    predict(net,test_loader)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




