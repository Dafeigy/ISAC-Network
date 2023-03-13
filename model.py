import torch.nn as nn
import torch
import cv2
import numpy as np
import torch.nn.functional as F



class Net3d(nn.Module):
    '''
    Network structure in paper: Multi-sensor System for Driver’s Hand-Gesture Recognition.
    '''
    def __init__(self) -> None:
        super(Net3d, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels = 3, out_channels=25, kernel_size = 5, stride = 2),
            nn.Conv3d(in_channels= 25,out_channels = 25, kernel_size = 5, stride = 2)
        )
        self.flatten = nn.Flatten()
        self.stage2 = nn.Sequential(
            nn.Linear(7500, 1024),
            nn.Linear(1024,128),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        x = self.stage1(x)
        print(x.shape)
        x = x.view(-1)
        x = self.stage2(x)
        return F.softmax(x, dim=0)
    


class MTNet(nn.Module):
    '''
     Modality Translation Network in DensePose From WiFi.
     
     `Input`: [150 x 3 x 3]
     `Output`: [3 x 720 x 1280]
    '''
    def __init__(self) -> None:
        super(MTNet, self).__init__()
        self.f_fusion = nn.Flatten()
        self.FT1 = self.build_head_encoder()
        self.FT2 = self.build_head_encoder()

    def build_head_encoder(self,hidden_dim=1024, output_dim=512):
        '''
        hidden_dim and output_dim are hyper params default as classical num of 1024 and 512.
        '''
        encoder = nn.Sequential(
            nn.Linear(1350,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
            nn.ReLU()
        )
        return encoder


if __name__ == '__main__':
    # Test image in/output
    # input buffer size: [60,3,32,32]
    model = Net3d()
    input_img = torch.rand(3,60,32,32)
    out = model(input_img)
    class_index = torch.argmax(out).data.item()


    

