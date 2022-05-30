from torch import nn
import torch 
import torch.nn.functional as F
from torch import cat 

class TenConv(nn.Module):
    """
    our first proposed architecture with 10 convolutional layer
    """
    def __init__(self) -> None:
        super().__init__() 

        n_channel  = 64                  # number of output channels 


        # convolutional layers for encoder
        self.enc_conv0 = nn.Conv2d(3, n_channel, kernel_size=2, stride=2)
        self.enc_conv1 = nn.Conv2d(n_channel, n_channel, kernel_size=2, stride=2)
        self.enc_conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=2, stride=2)
        self.enc_conv3 = nn.Conv2d(n_channel, n_channel, kernel_size=2, stride=2)  
        self.enc_conv4 = nn.Conv2d(n_channel, n_channel, kernel_size=2, stride=2)  

        # transpose convolutional layers for encoder
        self.dec_conv0 = nn.ConvTranspose2d(n_channel, n_channel, kernel_size=2, stride=2)
        self.dec_conv1 = nn.ConvTranspose2d(n_channel*2, n_channel, kernel_size=2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(n_channel*2, n_channel, kernel_size=2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(n_channel*2, n_channel, kernel_size=2, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d(n_channel*2, 3, kernel_size=2, stride=2)
    
        # Batch normalization blocks 
        self.batch_norm0 = nn.BatchNorm2d(n_channel)
        self.batch_norm1 = nn.BatchNorm2d(n_channel)
        self.batch_norm2 = nn.BatchNorm2d(n_channel)
        self.batch_norm3 = nn.BatchNorm2d(n_channel)
        self.batch_norm4 = nn.BatchNorm2d(n_channel)
        self.batch_norm5 = nn.BatchNorm2d(n_channel)
        self.batch_norm6 = nn.BatchNorm2d(n_channel)
        self.batch_norm7 = nn.BatchNorm2d(n_channel)
        self.batch_norm8 = nn.BatchNorm2d(n_channel)

        # Drop-out layers: commented because they decreased the validation PSNR
    '''
        self.drop_out0 = nn.Dropout2d(p=0.2)
        self.drop_out1 = nn.Dropout2d(p=0.2)
        self.drop_out2 = nn.Dropout2d(p=0.2)
        self.drop_out3 = nn.Dropout2d(p=0.2)
        self.drop_out4 = nn.Dropout2d(p=0.2)
        self.drop_out5 = nn.Dropout2d(p=0.2)
        self.drop_out6 = nn.Dropout2d(p=0.2)
        self.drop_out7 = nn.Dropout2d(p=0.2)
        self.drop_out8 = nn.Dropout2d(p=0.2)'''
      
        
    
    def forward(self, x):

        skips = [x]   # skip connection container

        # encoder
        x = F.relu(self.enc_conv0(x))
        x = self.batch_norm0(x)
        #  # x = self.drop_out0(x)
        skips.append(x)
        x = F.relu(self.enc_conv1(x))
        x = self.batch_norm1(x)
        # x = self.drop_out1(x)
        skips.append(x)
        x = F.relu(self.enc_conv2(x))
        x = self.batch_norm2(x)
        # x = self.drop_out2(x)
        skips.append(x)
        x = F.relu(self.enc_conv3(x))
        x = self.batch_norm3(x)
        # x = self.drop_out3(x)
        skips.append(x)
        x = F.relu(self.enc_conv4(x))
        x = self.batch_norm4(x)
        # x = self.drop_out4(x)

        # decoder
        x = F.relu(self.dec_conv0(x))
        x = self.batch_norm5(x)
        # x = self.drop_out5(x)
        x = cat((x, skips.pop()), axis=1)
        x = F.relu(self.dec_conv1(x))
        x = self.batch_norm6(x)
        # x = self.drop_out6(x)
        x = cat((x, skips.pop()), axis=1)
        x = F.relu(self.dec_conv2(x))
        x = self.batch_norm7(x)
        # x = self.drop_out7(x)
        x = cat((x, skips.pop()), axis=1)
        x = F.relu(self.dec_conv3(x))
        x = self.batch_norm8(x)
        # x = self.drop_out8(x)
        x = cat((x, skips.pop()), axis=1)
        x = torch.sigmoid(self.dec_conv4(x))

        return x
