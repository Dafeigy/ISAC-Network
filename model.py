import torch.nn as nn
import torch
import cv2
import numpy as np
import torch.nn.functional as F



class Net3d(nn.Module):
    '''
    Network structure in paper: Multi-sensor System for Driver’s Hand-Gesture Recognition.
    ## To use:
    ```
    # Test image in/output
    # input buffer size: [3,60,32,32]
    model = Net3d()
    input_img = torch.rand(3,60,32,32)
    out = model(input_img)
    class_index = torch.argmax(out).data.item()
    ```
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
            nn.Linear(128, 12)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = x.view(-1)
        x = self.stage2(x)
        return F.softmax(x, dim=0)
    

class mmWave(nn.Module):
    '''
    RDI input: [4 x 32 x 32]
    RAI input: [1 x 32 x 32]

    To use:
    ```
    model = mmWave()
    input1 = torch.rand(4,32,32).unsqueeze(0)   # RDI Data
    input2 = torch.rand(1,32,32).unsqueeze(0)   # RAI Data
    output= model(input1,input2)
    class_index = torch.argmax(output).data.item()
    print(output)
    print(class_index)
    ```
    '''
    def __init__(self) -> None:
        super(mmWave, self).__init__()
        self.RDI = self.build_branch(RDI = True)
        self.RAI = self.build_branch(RDI = False)
        self.RDI_Drop = nn.Dropout()
        self.RAI_Drop = nn.Dropout()
        self.dense = nn.Linear(1024,12)
        
    def build_branch(self,RDI=True):
        feature_head = nn.Sequential(
            self.build_conv_block(in_channel=4 if RDI else 1,out_channel=32),
            self.build_conv_block(in_channel=32,out_channel=64,prob_drop=0.4),
            self.build_conv_block(in_channel=64,out_channel=128,prob_drop=0.4),
            nn.Flatten(),
            self.build_dense_block(in_channel=86528),
            self.build_dense_block(),
            nn.LSTM(512,512,1),
        )
        return feature_head

    def build_conv_block(self, in_channel, out_channel,prob_drop=0):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, stride=1,kernel_size=3),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Dropout(p = prob_drop)
        )
        return conv_block
    
    def build_dense_block(self,in_channel=512, out_channel=512):
        dense_block = nn.Sequential(
            nn.Linear(in_channel,out_channel),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        return dense_block
    
    def forward(self,rdi,rai):
        RDI_, (ht1,ct1) = self.RDI(rdi)
        RAI_, (ht1,ct1) = self.RAI(rai)
        RDI_ = self.RDI_Drop(RDI_)
        RAI_ = self.RAI_Drop(RAI_)
        fusion_feature =torch.concat((RAI_,RDI_),dim=1)
        
        logits = self.dense(fusion_feature)
        return F.softmax(logits.squeeze(0),dim=0)

class MTNet(nn.Module):
    '''
    Modality Translation Network in DensePose From WiFi.
     
     `Input`: [150 x 3 x 3]
     `Output`: [3 x 720 x 1280]
    '''
    def __init__(self) -> None:
        super(MTNet, self).__init__()
        self.FT1 = self.build_head_encoder()
        self.FT2 = self.build_head_encoder()
        self.fusion_block = self.build_fusion_blcok()

    def build_fusion_block(self):
        fusion_block = nn.Sequential(
            nn.Linear(1024,576),
            nn.ReLU()
        )
        return fusion_block
    
    def build_unet(self):
        unet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2),
        )

    def build_head_encoder(self,hidden_dim=1024, output_dim=512):
        '''
        hidden_dim and output_dim are hyper params default as classical num of 1024 and 512.
        '''
        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1350,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
            nn.ReLU()
        )
        return encoder


if __name__ == '__main__':
    # Test image in/output
    # input buffer size: [60,3,32,32]
    # model = Net3d()
    # input_img = torch.rand(3,60,32,32)
    # out = model(input_img)
    # class_index = torch.argmax(out).data.item()

    model = mmWave()
    print(model)
    input1 = torch.rand(4,32,32).unsqueeze(0)
    input2 = torch.rand(1,32,32).unsqueeze(0)
    output= model(input1,input2)
    class_index = torch.argmax(output).data.item()
    print(output)
    print(class_index)


    

