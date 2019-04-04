#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser('pix4d')

## general settings.
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--batch', type=int, default=1) #batch size
parser.add_argument('--num_epoch', type=int, default=5000) #number of epochs
config, _ = parser.parse_known_args()


# In[15]:


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


# In[16]:


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


# In[ ]:


"""
Training
"""

def train_model(net, train_loader,criterion,num_epoch,optimizer):
    print('<training start>')
    writer_loss = SummaryWriter('./log/loss')
    for epoch in range(num_epoch):
        since = time.time()
        running_train_loss = 0.0
        for i, (imgs,labels) in enumerate(train_loader):
            
            if config.use_gpu:
                imgs = Variable(imgs.cuda()).float()
                labels = Variable(labels.cuda()).float()

            else:
                imgs = Variable(imgs).float()
                labels = Variable(labels).float()

            optimizer.zero_grad()
            out = net(imgs)
            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        if epoch!= 0:
                epoch_time=time.time()-since
                print("Time:{:f} Epoch [{}/{}] Train: {:.4f}".format(
                                epoch_time,
                                epoch,
                                num_epoch,
                                float(running_train_loss)/train_num
                                ))
                
                writer_loss.add_scalar('Loss', float(running_train_loss)/train_num, epoch)

        if epoch%5==0:
                # save trained model
                torch.save(net.state_dict(), "trained_model/Net_" + str(epoch) + ".pkl")

        if epoch%5==0:    
                image = out.max(1)[1].detach()
                if config.use_gpu:
                    image = image.cpu().data.numpy().squeeze()
                else:
                    image = image.data.numpy().squeeze()
                cv2.imwrite('intermediate_results/'+str(epoch)+'.png',255*image)
    writer_loss.close()       
    print
    print('<Finished Training>')
    return 0



if __name__ == '__main__':
    path_dic={
        "train":'images',
        "val": 'images'
    }

    """
    Data loading
    """
    train = Pix4dDataloader(path_dic["train"])
    train_loader = torch.utils.data.DataLoader(train, batch_size=config.batch, num_workers=1)
    train_num = train.__len__()

    if config.use_gpu:
        net=Net().cuda()
    else:
        net=Net()
    
    lr = 0.0003
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    print(net)
    print("lr:",lr)
    start_time=time.time()
    train_model(net,train_loader,criterion, num_epoch=config.num_epoch,optimizer=optimizer)



# In[ ]:




