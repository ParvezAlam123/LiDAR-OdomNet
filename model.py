import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np 
import struct 
import time 



class Attention(nn.Module):
    def __init__(self, dim=1792):
        super().__init__()
        
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2*dim, dim)
        
        self.bn = nn.BatchNorm2d(64) 

        
        
    def forward(self, current_feature, past_feature):
        B, C, H, dim = current_feature.shape 
        
        scale = np.sqrt(2 * dim)
        
        Query = self.fc1(current_feature)
        Key = Value = self.fc1(past_feature)
        
        weight = (Query @ Key.transpose(2,3)) / scale 
        weight = F.softmax(weight, dim=-1)
        attention_feature = weight @ Value  

        feature = Query + attention_feature          # residual connection 

        feature = F.relu(self.fc2(feature))
        feature = self.bn(feature)

        return feature 





class Encoder(nn.Module):
    def __init__(self):
        super().__init__() 
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = F.relu(self.conv4(x)) 
        return out 
    


class Model(nn.Module):
    def __init__(self):
        super().__init__() 

        self.past_encoder = Encoder()
        self.current_encoder = Encoder() 
        self.attention_block = Attention() 

        self.fct1 = nn.Linear(100352 * 2, 1024)
        self.fct2 = nn.Linear(1024, 512)
        self.fct3 = nn.Linear(512, 128)
        self.fct4 = nn.Linear(128, 64)
        self.fct5 = nn.Linear(64, 3)

        self.fcr1 = nn.Linear(100352* 2, 1024)
        self.fcr2 = nn.Linear(1024, 512)
        self.fcr3 = nn.Linear(512, 128)
        self.fcr4 = nn.Linear(128, 64)
        self.fcr5 = nn.Linear(64, 9)



    def forward(self, current_range_img, past_range_img):
        B, C, H, dim = current_range_img.shape
        current_feature = self.current_encoder(current_range_img)
        past_feature = self.past_encoder(past_range_img)

        feature = self.attention_block(current_feature, past_feature) 
        
       

        # maxpool in channel dimension 
        feature = feature.max(dim=1, keepdim=True)[0].reshape(B, 1, -1)
        current_feature = current_feature.max(dim=1, keepdim=True)[0].reshape(B, 1, -1)
       

        # concatenate current feature and feature after attention block 
        feature = torch.cat((feature, current_feature), dim=2)

        # regress translation values 
        translation_feature = F.relu(self.fct1(feature))
        translation_feature = F.relu(self.fct2(translation_feature))
        translation_feature = F.relu(self.fct3(translation_feature))
        translation_feature = F.relu(self.fct4(translation_feature))
        translation_output = self.fct5(translation_feature) 

        # regress rotation quaternion values 
        rotation_feature = F.relu(self.fcr1(feature))
        rotation_feature = F.relu(self.fcr2(rotation_feature))
        rotation_feature = F.relu(self.fcr3(rotation_feature))
        rotation_feature = F.relu(self.fcr4(rotation_feature))
        rotation_output = self.fcr5(rotation_feature)

        return translation_output, rotation_output 
    


    

        

        