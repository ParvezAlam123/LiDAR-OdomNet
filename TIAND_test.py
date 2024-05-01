import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np 
import struct 
import time 
import math 
from dataset import KITTI_ODOM_DATA, CustomData, TIAND 
from model import Model
import matplotlib.pyplot as plt  
from torchsummary import summary 




pcd_scene_path = "/media/parvez_alam/Expansion/ICRA_2024_TIAND/2nd_SEPT/Lidar_compressed_2nd_Sept/scene4"
gnns_path = "/media/parvez_alam/Expansion/ICRA_2024_TIAND/2nd_SEPT/GNSS_2nd_sept/scene4-novatel-oem7-inspva.csv"


valid_ds = TIAND(pcd_scene_path=pcd_scene_path,gnns_path=gnns_path)
valid_loader = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





model = Model()
model.to(device)







model.load_state_dict(torch.load("model_state_current_feature.pth"))
model.to(device)
model.eval() 










optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 

training_loss = [] 








    






def val(model, valid_loader):
    for n, data in enumerate(valid_loader):
       if n == 10:
           current_range_img = data['current_range_img'].to(device).float().unsqueeze(dim=1)
           past_range_img = data['past_range_img'].to(device).float().unsqueeze(dim=1)
           current_latitude = data["current_latitude"].to(device).float()
           current_longitude = data["current_longitude"].to(device).float().item() 
           past_latitude  = data["past_latitude"].to(device).float().item() 
           past_longitude = data["past_longitude"].to(device).float().item() 
           current_height = data["current_height"].to(device).float().item() 
           past_height = data["past_height"].to(device).float().item()

           #relative_pose = data['relative_pose'].to(device).float()
           translation_output, rotation_output =  model(current_range_img, past_range_img) 
            
        
          



           predicted_displacement = math.sqrt((translation_output[0, 0, 0].item())**2 + (translation_output[0, 0, 1].item())**2)

           a = 6378.137 * 1000  # km to meter 
           e_square = 0.00669437999 
           R_current = a / math.sqrt(1 - e_square * math.sin(current_latitude *(np.pi/180))**2) 
           R_past = a / math.sqrt(1 - e_square * math.sin(past_latitude * (np.pi/180))**2)

           current_x = (R_current + current_height) * math.cos(current_latitude * (np.pi/180)) * math.cos(current_longitude * (np.pi/180))
           current_y = (R_current + current_height) * math.cos(current_latitude * (np.pi / 180)) * math.sin(current_longitude * (np.pi/180))


           past_x = (R_past + past_height) * math.cos(past_latitude * (np.pi/180)) * math.cos(past_longitude * (np.pi/180))
           past_y = (R_past + past_height) * math.cos(past_latitude * (np.pi / 180)) * math.sin(past_longitude * (np.pi/180)) 

           gt_displacement = math.sqrt((current_x - past_x)**2 + (current_y - past_y)**2) 

        

           print("prediction = ", predicted_displacement, "gt = ", gt_displacement)
        
       else:
           continue 
       

       
        


val(model, valid_loader)












