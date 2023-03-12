import torch.nn as nn
import torch
import cv2
import numpy as np
import torch.nn.functional as F



class Net3d(nn.Module):
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
    

        

if __name__ == '__main__':
    # Test image in/output
    # input buffer size: [60,3,32,32]
    model = Net3d()
    input_img = torch.rand(3,60,32,32)
    out = model(input_img)
    class_index = torch.argmax(out).data.item()


    

