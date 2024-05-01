import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np 
import struct 
import time 
import math 
import open3d as o3d 
import pandas as pd







def ProjectPCimg2SphericalRing(PC, H_input = 64, W_input = 1800):
    
    num_points, dim = PC.shape

    degree2radian = math.pi / 180
    nLines = H_input    
    AzimuthResolution = 360.0 / W_input # degree
    FODown = -(-24.8) # take absolute value
    FOUp = 2.0
    FOV = FODown+FOUp
    
    
    range_img = np.zeros((H_input, W_input), dtype=np.float32)
    for i in range(num_points):
        x = PC[i][0]
        y = PC[i][1]
        z = PC[i][2]
        r = math.sqrt(x**2+y**2+z**2)
        
        pitch = math.asin(z / r) * (180/np.pi)
        yaw = math.atan2(y, x)   * (180/np.pi)
        if pitch < -24.8:
            pitch = -24.8
        
        
        if np.isnan(64 * ((FOUp - pitch)/FOV)):
            u = 63
        else:
           u = int(64 * ((FOUp - pitch)/FOV)) - 1 
        v = int(1800 * ((yaw+180)/360))  - 1 

        
        range_img[u][v] = r 

    return range_img

    

class KITTI_ODOM_DATA(Dataset):
    def __init__(self, data_path, calib_path, pose_path, x_min = 0, x_max=70, y_min=-20, y_max=20, z_min=-20, z_max=20, train=True):
        
        
        self.train = train 
        self.x_min = x_min 
        self.x_max = x_max 
        self.y_min = y_min 
        self.y_max = y_max 
        self.z_min = z_min 
        self.z_max = z_max 

        
        self.data_path = data_path
        self.calib_path = calib_path 
        self.pose_path = pose_path 
        
        if train == True:
            data_seq = sorted(os.listdir(self.data_path))[0:7]
            calib_seq = sorted(os.listdir(self.calib_path))[0:7]
            pose_seq = sorted(os.listdir(self.pose_path))[0:7]
        else:
            data_seq = sorted(os.listdir(self.data_path))[10:11]
            calib_seq = sorted(os.listdir(self.calib_path))[10:11]
            pose_seq = sorted(os.listdir(self.pose_path))[10:11]
            
        self.files = [] 
        
        for i in data_seq:
            data_seq_path = os.path.join(self.data_path, i, 'velodyne')
            calib_seq_file_path = os.path.join(self.calib_path, i,  'calib.txt')
            pose_seq_file_path = os.path.join(self.pose_path, i+'.txt')
            
            with open(calib_seq_file_path, 'r') as f:
                lines = f.readlines()
                Tr = np.array(lines[4].strip('\n').split()[1:], dtype=np.float32).reshape(3,4)
                velo_to_cam = np.zeros((4,4), dtype=np.float32)
                velo_to_cam[:3, :] = Tr
                velo_to_cam[3, 3] = 1  
            
            # store poses for every frame 
            poses = [] 
            with open(pose_seq_file_path, 'r') as f:
                lines = f.readlines() 
                for l in lines:
                    pose = l.strip('\n').split() 
                    poses.append(pose)
                    
            #global past_frame, past_pose        
            past_frame = 0
            past_pose = 0
            frame_list = sorted(os.listdir(data_seq_path)) 
            
            for n in range(len(frame_list)):
                sample = {} 
                
                if n % 2 == 0:   # past frame 
                    past_frame = os.path.join(data_seq_path, frame_list[n])
                    frame_temp_pose = np.array(poses[n], dtype=np.float32).reshape(3,4)
                    frame_pose = np.zeros((4,4), dtype=np.float32)
                    frame_pose[:3, :] = frame_temp_pose 
                    frame_pose[3,3] = 1 
                    past_pose = np.dot(frame_pose, velo_to_cam)  # get velo pose in global coordinate system
                else:
                    current_frame = os.path.join(data_seq_path, frame_list[n]) 
                    frame_temp_pose = np.array(poses[n], dtype=np.float32).reshape(3,4)
                    current_pose = np.zeros((4,4), dtype=np.float32) 
                    current_pose[:3, :] = frame_temp_pose 
                    current_pose[3,3] = 1 
                    current_pose = np.dot(current_pose, velo_to_cam) # get velo pose in global coordinate system
                    
                    sample['past_frame'] = past_frame
                    sample['past_pose'] = past_pose 
                    sample['current_frame'] = current_frame 
                    sample['current_pose'] = current_pose 
                    
                    self.files.append(sample)
                    
                    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        past_frame_path = self.files[index]['past_frame'] 
        past_pose = self.files[index]['past_pose']
        current_frame_path = self.files[index]['current_frame']
        current_pose = self.files[index]['current_pose']

        
        # read past pcd data 
        past_pcd = [] 
        size_float = 4
        with open(past_frame_path, 'rb') as f:
            byte = f.read(size_float * 4)
            while byte:
                x,y,z, intensity = struct.unpack("ffff", byte)
                past_pcd.append([x,y,z])
                byte = f.read(size_float * 4)
        
        past_pcd = np.array(past_pcd)
        
        # read current pcd data 
        current_pcd = [] 
        with open(current_frame_path, 'rb') as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                current_pcd.append([x,y,z])
                byte = f.read(size_float * 4) 
        
        current_pcd = np.array(current_pcd)      
        
        

        # get current range image 
        current_range_img = ProjectPCimg2SphericalRing(current_pcd)

        # get past range image 
        past_range_img = ProjectPCimg2SphericalRing(past_pcd)


        # get the relative pose of second with respect to first frame
        inv_first_pose = np.linalg.inv(past_pose)
        relative_pose = np.dot(inv_first_pose, current_pose)
        

        
        return {'current_range_img':current_range_img, 'past_range_img':past_range_img, 'relative_pose': relative_pose}
    
    
      




class CustomData(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path 

        self.files = [] 

        pcd_files = sorted(os.listdir(self.folder_path)) 
        i = 0 
        while i+1 < len(pcd_files):
            past_pcd_path = os.path.join(self.folder_path, pcd_files[i]) 
            current_pcd_path = os.path.join(self.folder_path, pcd_files[i+1])
            sample = {}
            sample["past_pcd_path"] = past_pcd_path 
            sample["current_pcd_path"] = current_pcd_path
            self.files.append(sample)
            
            i = i + 2 



    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        past_frame_path = self.files[index]["past_pcd_path"]
        current_frame_path = self.files[index]["current_pcd_path"]

        # read past pcd data 
        past_pcd = o3d.io.read_point_cloud(past_frame_path) 
        past_pcd = np.asarray(past_pcd.points) 

        
     
        # read current pcd data 
        
        current_pcd = o3d.io.read_point_cloud(current_frame_path)
        current_pcd = np.asarray(current_pcd.points) 


        # get current range image 
        current_range_img = ProjectPCimg2SphericalRing(current_pcd)

        # get past range image 
        past_range_img = ProjectPCimg2SphericalRing(past_pcd) 


        return {'current_range_img':current_range_img, 'past_range_img':past_range_img}
    






class TIAND(Dataset):
    def __init__(self, pcd_scene_path, gnns_path):
        self.pcd_scene_path = pcd_scene_path 
        self.gnns_path = gnns_path 

        self.files = [] 
        
        self.pcds = sorted(os.listdir(self.pcd_scene_path)) 
        df = pd.read_csv(self.gnns_path)
        latitude = df[".latitude"]
        longitude = df[".longitude"]
        height = df[".height"]
        j = 0
        i = 0
        while i < len(self.pcds):
                past_pcd_path = os.path.join(self.pcd_scene_path, self.pcds[i])
                past_latitude_value = float(latitude[j])
                past_longitude_value = float(longitude[j])
                past_height_value = float(height[j])
                j = j + 5

                current_pcd_path = os.path.join(self.pcd_scene_path, self.pcds[i+1])
                current_latitude_value = float(latitude[j])
                current_longitude_value = float(longitude[j])
                current_height_value = float(height[j])
                i = i + 2 
                j = j + 5
                sample = {} 
                sample["past_frame_path"] = past_pcd_path
                sample["past_latitude_value"] = past_latitude_value 
                sample["past_longitude_value"] = past_longitude_value 
                sample["past_height_value"] = past_height_value
                sample["current_frame_path"] = current_pcd_path
                sample["current_latitude_value"] = current_latitude_value 
                sample["current_longitude_value"] = current_longitude_value 
                sample["current_height_value"] = current_height_value
                self.files.append(sample)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        current_frame_path = self.files[index]["current_frame_path"]
        current_latitude = self.files[index]["current_latitude_value"]
        current_longitude = self.files[index]["current_longitude_value"]
        current_height = self.files[index]["current_height_value"]

        past_frame_path = self.files[index]["past_frame_path"]
        past_latitude = self.files[index]["past_latitude_value"]
        past_longitude = self.files[index]["past_longitude_value"]
        past_height = self.files[index]["past_height_value"]

        
        # read past pcd data 
        past_pcd = o3d.io.read_point_cloud(past_frame_path) 
        past_pcd = np.asarray(past_pcd.points) 

        
     
        # read current pcd data 
        
        current_pcd = o3d.io.read_point_cloud(current_frame_path)
        current_pcd = np.asarray(current_pcd.points) 


        # get current range image 
        current_range_img = ProjectPCimg2SphericalRing(current_pcd)

        # get past range image 
        past_range_img = ProjectPCimg2SphericalRing(past_pcd) 
        
     

        return {'current_range_img':current_range_img, 'past_range_img':past_range_img, "current_latitude":current_latitude, "past_latitude":past_latitude,
                "current_longitude":current_longitude, "past_longitude":past_longitude, "past_height":past_height, "current_height":current_height}
    
        
            
        

        
    
     


        
    
