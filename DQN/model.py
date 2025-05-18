import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_shape, num_actions) -> None: 
        super(DQN, self).__init__()

        height, width, in_channels = state_shape 

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)

        def conv_output_size(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
            from math import floor
            if type(kernel_size) is not tuple:
                kernel_size = (kernel_size, kernel_size)
            if type(stride) is not tuple:
                stride = (stride, stride)
            if type(padding) is not tuple:
                padding = (padding, padding)
            h = floor(((h_w[0] + (2 * padding[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
            w = floor(((h_w[1] + (2 * padding[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
            return h, w

        conv1_out_h, conv1_out_w = conv_output_size((height, width), kernel_size=8, stride=4)
        pool1_out_h, pool1_out_w = conv_output_size((conv1_out_h, conv1_out_w), kernel_size=2, stride=2)
        conv2_out_h, conv2_out_w = conv_output_size((pool1_out_h, pool1_out_w), kernel_size=4, stride=3)
        pool2_out_h, pool2_out_w = conv_output_size((conv2_out_h, conv2_out_w), kernel_size=2, stride=2)
        
        flattened_size = 64 * pool2_out_h * pool2_out_w 

        self.fc1 = nn.Linear(flattened_size, 256) 
        self.fc_out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) 
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        q_values = self.fc_out(x) 
        return q_values
