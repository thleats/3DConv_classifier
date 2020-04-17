

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import glob
from torch.utils import data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import random
import statistics
from scipy.ndimage.interpolation import rotate
from zipfile import ZipFile

zippath = '/content/024_0_143.zip'


with ZipFile(zippath, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

path = '/content/'
current = os.getcwd()
os.chdir(path)
import pdb

files_train=[]
files_test=[]
for file in glob.glob("*.npy"):
    temp=file.split('_')
    if int(temp[1])==-1:
      files_test.append(file)
    else:
      files_train.append(file)
os.chdir(current)

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.files=data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)
        #return 100

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        filename=self.files[index]

        ID=int(filename[0:3])
        array = np.load(filename)
        num=random.uniform(0,1)
        if num<0.5:
            out= np.flipud(array).copy()
        else:
            out = array
            
        num=random.uniform(0,1)
        if num<0.5:
            out2= np.fliplr(out).copy()
        else:
            out2 = out
            
        num2=round(random.uniform(-5,5))
        out3= np.roll(out2,num2,axis=0).copy()
        
        num2=round(random.uniform(-5,5))
        out4= np.roll(out3,num2,axis=1).copy()
        
        num2=round(random.uniform(-1,1))
        out5= np.roll(out4,num2,axis=2).copy()
        
        num2=round(random.uniform(-1,1))
        out6= np.rot90(out5,k=num2).copy()
        
        return np.asarray(ID).astype('float32'), np.asarray(out6).astype('float32')

class ConvNetwork(nn.Module):
  def __init__(self):
    super(ConvNetwork,self).__init__()
    Conv3d=nn.Conv3d
    self.c1=Conv3d(1,2,(2,2,3),stride=(2,2,1),padding=(0,0,1))#go from 100,100,5 to 50,50,5
    self.c2=Conv3d(2,5,(2,2,3),stride=(2,2,1),padding=(0,0,1))#go from 50,50,5 to 25,25,5
    self.c3=Conv3d(5,10,(3,3,2),stride=(2,2,1),padding=(0,0,0))#go from 25,25,5 to 12,12,4
    self.c4=Conv3d(10,25,(2,2,2),stride=(2,2,2),padding=(0,0,0))#go from 12,12,4 to 6,6,2
    self.c5=Conv3d(25,25,(6,6,2),stride=(1,1,2),padding=(0,0,0))#go from 6,6,2 to 1,1,1
    self.activation = nn.ReLU()
    self.bn1=torch.nn.BatchNorm3d(2)
    self.bn2=torch.nn.BatchNorm3d(5)
    self.bn3=torch.nn.BatchNorm3d(10)
    self.bn4=torch.nn.BatchNorm3d(25)
    
    
  def forward(self,x):
    x1=self.bn1(self.activation(self.c1(x)))
    x2=self.bn2(self.activation(self.c2(x1)))
    x3=self.bn3(self.activation(self.c3(x2)))
    x4=self.bn4(self.activation(self.c4(x3)))
    x5=(self.c5(x4))
    

    return x5

epochs=400
#define the objective function
objective = torch.nn.CrossEntropyLoss()
model = ConvNetwork()
model = model.cuda()

optimizer = optim.Adam(model.parameters(),lr=1e-4)


params={'batch_size':500,'shuffle':True}
training_set=Dataset(files_train)
test_set=Dataset(files_test)
training_generator=data.DataLoader(training_set,**params)
test_generator=data.DataLoader(test_set,**params)
loop = tqdm(total=epochs, position = 0)
loss_record_train=[]
loss_temp_train=[]
loss_record_test=[]
loss_temp_test=[]
accuracy = []
for i in range(epochs):
    loss_temp_train=[]
    loss_temp_test=[]
    acc_temp=[]
    for things in training_generator:
        
        array = things[1].unsqueeze(1)
        y = things[0]
        optimizer.zero_grad()
        
        yhat = model(array.cuda()).squeeze(-1).squeeze(-1).squeeze(-1)
        #print(yhat.size())

        loss = objective(yhat, y.long().cuda())
        loss.backward()
        optimizer.step()
        loss_temp_train.append(loss.item())
    with torch.no_grad():
      for things in test_generator:
          
          array = things[1].unsqueeze(1)
          y = things[0]
          
          yhat = model(array.cuda()).squeeze(-1).squeeze(-1).squeeze(-1)
          #print(yhat.size())

          loss = objective(yhat, y.long().cuda())
          loss_temp_test.append(loss.item())
          temp=torch.argmax(yhat,dim=1)-y.cuda()
          acc_temp.append(1-(len(torch.nonzero(temp))/len(temp)))
    loss_record_train.append(statistics.mean(loss_temp_train))
    accuracy.append(statistics.mean(acc_temp))
    loss_record_test.append(statistics.mean(loss_temp_test))
    print(statistics.mean(acc_temp))
    loop.update(1)
    
plt.plot(loss_record_train)
plt.plot(loss_record_test)
plt.plot(accuracy)

array.size()

plt.plot(loss_record_train)
plt.plot(loss_record_test)
plt.plot(accuracy)
