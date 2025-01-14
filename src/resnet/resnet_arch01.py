import torch 
import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=8, num_antennas=3, num_subcarriers=30, num_time_slices=10):
        super(CustomResNet18, self).__init__()
        
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.num_time_slices = num_time_slices
        
        # Load the standard ResNet18
        self.resnet18 = models.resnet18(pretrained=False)
        
        # Modify the first convolutional layer to accept 2 input channels (amplitude and phase)
        self.resnet18.conv1 = nn.Conv2d(
            in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Modify the fully connected layer to output the number of classes
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        x: [batch_size, num_time_slices, num_antennas*num_subcarriers*2]
        """
        num_antennas = self.num_antennas
        num_subcarriers = self.num_subcarriers

        amplitudes = x[:,:,:num_antennas*num_subcarriers]
        phases = x[:,:,num_antennas*num_subcarriers:]

        # Resulting shape: [batch_size, time_slices, num_antennas * num_subcarriers, 2]
        csi_data = torch.stack((amplitudes, phases), dim=-1)
        
        # Resulting shape: [batch_size, 2, num_antennas*num_subcarriers, time_slices]
        csi_data = csi_data.permute(0,3,2,1)
        # Forward through ResNet18
        return self.resnet18(csi_data)   
