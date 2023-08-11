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
from dataset import KITTI_ODOM_DATA
from model import Model
import matplotlib.pyplot as plt  



data_path = "/media/parvez_alam/Expansion/Kitti/Odometry/data_odometry_velodyne/dataset/sequences"
calib_path = "/media/parvez_alam/Expansion/Kitti/Odometry/data_odometry_calib/dataset/sequences"
pose_path = "/media/parvez_alam/Expansion/Kitti/Odometry/data_odometry_poses/dataset/poses" 


#train_ds = KITTI_ODOM_DATA(data_path, calib_path, pose_path) 
#train_loader = DataLoader(dataset= train_ds, batch_size=8, shuffle=True)

valid_ds = KITTI_ODOM_DATA(data_path, calib_path, pose_path, train=False) 
valid_loader = DataLoader(dataset = valid_ds, batch_size=1, shuffle=True)







device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



#loaded_checkpoint = torch.load("OdomNet3.pth") 
#model_parameters = loaded_checkpoint["model_state"]
#torch.save(model_parameters, "model_state.pth")

model = Model()
model.load_state_dict(torch.load("model_state.pth"))
model.to(device)
model.eval() 








optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 

training_loss = [] 


def train(model, train_loader, epoch):
    for i in range(epoch):
        running_loss = 0.0
        for n, data in enumerate(train_loader):
            current_range_img = data['current_range_img'].to(device).float().unsqueeze(dim=1)
            past_range_img = data['past_range_img'].to(device).float().unsqueeze(dim=1)
            relative_pose = data['relative_pose'].to(device).float()
            translation_output, rotation_output =  model(current_range_img, past_range_img)
        
            B, Row, Col = relative_pose.shape 

            rotation_target = relative_pose[:, 0:3, 0:3].reshape(B, 1, -1)
            translation_target = relative_pose[:, :3, 3:4].reshape(B, 1, -1)  
        

            # calculate translation loss 
            loss_trans = (translation_output - translation_target)**2 
            loss_rot = (rotation_output - rotation_target)**2 
            loss = loss_trans.sum() + loss_rot.sum()  

            # backprop 
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()


            running_loss = running_loss + loss.item()

        training_loss.append(running_loss)
        print("running_loss = {}, epoch={}".format(running_loss, i+1))

        checkpoint={
            "epoch_number": i+1,
            "model_state": model.state_dict()
        }
        torch.save(checkpoint, "OdomNet3.pth")



R11 = []
R12 = [] 
R13 = [] 
R21 = []
R22 = [] 
R23 = [] 
R31 = [] 
R32 = [] 
R33 = []
tx = []
ty = []
tz = [] 

p_R11 = [] 
p_R12 = []
p_R13 = [] 
p_R21 = [] 
p_R22 = [] 
p_R23 = [] 
p_R31 = [] 
p_R32 = [] 
p_R33 = [] 
p_tx = []
p_ty = [] 
p_tz = [] 




def val(model, valid_loader):
    for n, data in enumerate(valid_loader):
        current_range_img = data['current_range_img'].to(device).float().unsqueeze(dim=1)
        past_range_img = data['past_range_img'].to(device).float().unsqueeze(dim=1)
        relative_pose = data['relative_pose'].to(device).float()
        translation_output, rotation_output =  model(current_range_img, past_range_img)  


        R11.append(relative_pose[0, 0, 0].item())
        R12.append(relative_pose[0, 0, 1].item())
        R13.append(relative_pose[0, 0, 2].item())
        R21.append(relative_pose[0, 1, 0].item())
        R22.append(relative_pose[0, 1, 1].item())
        R23.append(relative_pose[0, 1, 2].item())
        R31.append(relative_pose[0, 2, 0].item())
        R32.append(relative_pose[0, 2, 1].item())
        R33.append(relative_pose[0, 2, 2].item())
        tx.append(relative_pose[0, 0, 3].item())
        ty.append(relative_pose[0, 1, 3].item())
        tz.append(relative_pose[0, 2, 3].item())

        p_R11.append(rotation_output[0, 0, 0].item())
        p_R12.append(rotation_output[0, 0, 1].item())
        p_R13.append(rotation_output[0, 0, 2].item())
        p_R21.append(rotation_output[0, 0, 3].item())
        p_R22.append(rotation_output[0, 0, 4].item())
        p_R23.append(rotation_output[0, 0, 5].item())
        p_R31.append(rotation_output[0, 0, 6].item())
        p_R32.append(rotation_output[0, 0, 7].item())
        p_R33.append(rotation_output[0, 0, 8].item()) 
        p_tx.append(translation_output[0, 0, 0].item())
        p_ty.append(translation_output[0, 0, 1].item())
        p_tz.append(translation_output[0, 0, 2].item())



#train(model, train_loader, 80) 

val(model, valid_loader)


num_pose = np.arange(len(R11)) + 1


total_tx = [] 
total_ty = [] 
total_p_tx = [] 
total_p_ty = [] 

for i in range(len(tx)):
    if i == 0:
        total_tx.append(tx[i])
        total_ty.append(ty[i])
        total_p_tx.append(p_tx[i])
        total_p_ty.append(p_ty[i])
    else:
        total_tx.append(total_tx[i-1] + tx[i])
        total_ty.append(total_ty[i-1] + ty[i])
        total_p_tx.append(total_p_tx[i-1] + p_tx[i])
        total_p_ty.append(total_p_ty[i-1] + p_ty[i])


plt.plot(total_tx, total_ty, color='r', label='Trajectory (Ground Truth)')
plt.plot(total_p_tx, total_p_ty, color='g', label='Predicted Trajectory')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Trajectory plot for sequence 08 in XY plane")
plt.legend()
plt.show() 




# plot Rotation values curve 

"""plt.plot(num_pose[100:151], R11[100:151], color='r', label='R11')
plt.plot(num_pose[100:151], p_R11[100:151], color='g', label='predicted R11')
plt.xlabel("pose sequence")
plt.ylabel("R11 value")
plt.title("R11 Value Curve")
plt.legend()
plt.show() 

plt.plot(num_pose[100:151], R12[100:151], color='r', label='R12')
plt.plot(num_pose[100:151], p_R12[100:151], color='g', label="predicted R12")
plt.xlabel("pose sequence")
plt.ylabel("R12 value") 
plt.title("R12 Value Curve")
plt.legend() 
plt.show() 


plt.plot(num_pose[100:151], R13[100:151], color='r', label='R13')
plt.plot(num_pose[100:151], p_R13[100:151], color='g', label="predicted R13")
plt.xlabel("pose sequence")
plt.ylabel("R13 value")
plt.title("R13 Value Curve")
plt.legend() 
plt.show() 

plt.plot(num_pose[100:151], R21[100:151], color='r', label='R21') 
plt.plot(num_pose[100:151], p_R21[100:151], color='g', label='predicted R21')
plt.xlabel("pose sequence")
plt.ylabel("R21 value") 
plt.title("R21 Value Curve")
plt.legend() 
plt.show() 

plt.plot(num_pose[100:151], R22[100:151], color='r', label='R22')
plt.plot(num_pose[100:151], p_R22[100:151], color='g', label='predicted R22')
plt.xlabel("pose sequence")
plt.ylabel("R22 value")
plt.title("R22 Value Curve")
plt.legend() 
plt.show() 

plt.plot(num_pose[100:151], R23[100:151], color='r', label='R23')
plt.plot(num_pose[100:151], p_R23[100:151], color='g', label='predicted R23')
plt.xlabel("pose sequence")
plt.ylabel("R23 value")
plt.title("R23 Value Curve")
plt.legend() 
plt.show() 

plt.plot(num_pose[100:151], R31[100:151], color='r', label='R31')
plt.plot(num_pose[100:151], p_R31[100:151], color='g', label='predicted R31')
plt.xlabel("pose sequence")
plt.ylabel("R31 value")
plt.title("R31 Value Curve")
plt.legend() 
plt.show() 

plt.plot(num_pose[100:151], R32[100:151], color='r', label='R32')
plt.plot(num_pose[100:151], p_R32[100:151], color='g', label='predicted R32')
plt.xlabel("pose sequence")
plt.ylabel("R32 value")
plt.title("R32 Value Curve")
plt.legend() 
plt.show() 

plt.plot(num_pose[100:151], R33[100:151], color='r', label='R33')
plt.plot(num_pose[100:151], p_R33[100:151], color='g', label='predicted R33')
plt.xlabel("pose sequence")
plt.ylabel("R33 value")
plt.title("R33 Value Curve")
plt.legend() 
plt.show() 


# Plot translation values  curve 

plt.plot(num_pose[100:151], tx[100:151], color='r', label='tx')
plt.plot(num_pose[100:151], p_tx[100:151], color='g', label='predicted tx')
plt.xlabel("pose sequece")
plt.ylabel('tx value')
plt.title("tx Value Curve")
plt.legend() 
plt.show() 

plt.plot(num_pose[100:151], ty[100:151], color='r', label='ty')
plt.plot(num_pose[100:151], p_ty[100:151], color='g', label='predicted ty')
plt.xlabel("pose sequence")
plt.ylabel("ty value")
plt.title("ty Value Curve")
plt.legend() 
plt.show() 

plt.plot(num_pose[100:151], tz[100:151], color='r', label='tz')
plt.plot(num_pose[100:151], p_tz[100:151], color='g', label='predicted tz')
plt.xlabel("pose sequence")
plt.ylabel("tz value")
plt.title("tz Value Curve")
plt.legend() 
plt.show() 




# calculate error values 
# Rotation error degree 
R11_err = (np.array(R11) - np.array(p_R11))**2 
R12_err = (np.array(R12) - np.array(p_R12))**2 
R13_err = (np.array(R13) - np.array(p_R13))**2 
R21_err = (np.array(R21) - np.array(p_R21))**2 
R22_err = (np.array(R22) - np.array(p_R22))**2 
R23_err = (np.array(R23) - np.array(p_R23))**2 
R31_err = (np.array(R31) - np.array(p_R31))**2 
R32_err = (np.array(R32) - np.array(p_R32))**2 
R33_err = (np.array(R33) - np.array(p_R33))**2 
rotation_error = R11_err + R12_err + R13_err + R21_err + R22_err + R23_err + R31_err + R32_err + R33_err 
rotation_error = math.sqrt(np.sum(rotation_error) / len(R11)) 

# calculate the translation RMSE 

tx_error = (np.array(tx) - np.array(p_tx))**2 
ty_error = (np.array(ty) - np.array(p_ty))**2 
tz_error = (np.array(tz) - np.array(p_tz))**2 

translation_error = tx_error + ty_error + tz_error 
translation_error = math.sqrt(np.sum(translation_error) / len(tx)) 


print("RMSE Rotation = ", rotation_error)
print("RMSE translation = ", translation_error)"""








    


        







