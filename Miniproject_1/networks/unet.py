import torch
from torch import nn 
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self) -> None:
        super().__init__() 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.conv_pad = 'same'
        self.conv_stride = 1
        self.kernel_size = 3
        self.alpha = 0.1
        self.n = 3
        self.m = 3
        

        self.enc_conv0 = nn.Conv2d(self.n, 48, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.enc_conv5 = nn.Conv2d(48, 48, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.enc_conv6 = nn.Conv2d(48, 48, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)

        self.dec_conv5A = nn.Conv2d(96, 96, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv5B = nn.Conv2d(96, 96, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv4A = nn.Conv2d(144, 96, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv4B = nn.Conv2d(96, 96, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv3A = nn.Conv2d(144, 96, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv3B = nn.Conv2d(96, 96, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv2A = nn.Conv2d(144, 96, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv2B = nn.Conv2d(96, 96, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv1A = nn.Conv2d(96+self.n, 64, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv1B = nn.Conv2d(64, 32, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)
        self.dec_conv1C = nn.Conv2d(32, self.m, kernel_size=self.kernel_size, padding=self.conv_pad, stride=self.conv_stride, device=self.device)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)

        self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')


        
    def forward(self, x):
        
        x = x.to(self.device)
        # encoder
        skips = [x]
        
        x = F.leaky_relu(self.enc_conv0(x), self.alpha)
        x = F.leaky_relu(self.enc_conv1(x), self.alpha)
        x = self.pool1(x)
        skips.append(x)
        x = F.leaky_relu(self.enc_conv2(x), self.alpha)
        x = self.pool2(x)
        skips.append(x)
        x = F.leaky_relu(self.enc_conv3(x), self.alpha)
        x = self.pool3(x)
        skips.append(x)
        x = F.leaky_relu(self.enc_conv4(x), self.alpha)
        x = self.pool4(x)
        skips.append(x)
        x = F.leaky_relu(self.enc_conv5(x), self.alpha)
        x = self.pool5(x)
        x = F.leaky_relu(self.enc_conv6(x), self.alpha)

        # decoder
        x = self.upsample5(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv5A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv5B(x), self.alpha)
        x = self.upsample4(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv4A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv4B(x), self.alpha)
        x = self.upsample3(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv3A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv3B(x), self.alpha)
        x = self.upsample2(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv2A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv2B(x), self.alpha)
        x = self.upsample1(x)
        x = torch.cat((x, skips.pop()), axis=1)
        x = F.leaky_relu(self.dec_conv1A(x), self.alpha)
        x = F.leaky_relu(self.dec_conv1B(x), self.alpha)
        x = self.dec_conv1C(x)

        return x

if __name__ == '__main__':

    unet = UNet()
    print(unet)