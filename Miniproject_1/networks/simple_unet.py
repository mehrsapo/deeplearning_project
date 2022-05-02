from torch import nn 
import torch.nn.functional as F

class SUet(nn.Module):

    def __init__(self) -> None:
        super().__init__() 

        n_hidden = 64 

        self.enc_conv0 = nn.Conv2d(3, n_hidden, kernel_size=(5,5), stride=(1, 1))
        self.enc_conv1 = nn.Conv2d(n_hidden, n_hidden, kernel_size=(5,5), stride=(1, 1))
        self.enc_conv2 = nn.Conv2d(n_hidden, n_hidden, kernel_size=(4,4), stride=(2, 2))
        self.enc_conv3 = nn.Conv2d(n_hidden, 32, kernel_size=(4,4), stride=(1, 1)) 


        self.dec_conv0 = nn.ConvTranspose2d(32, n_hidden, kernel_size=(4,4), stride=(1, 1))
        self.dec_conv1 = nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=(4,4), stride=(2, 2))
        self.dec_conv2 = nn.ConvTranspose2d(n_hidden, n_hidden, kernel_size=(5,5), stride=(1, 1))
        self.dec_conv3 = nn.ConvTranspose2d(n_hidden, 3, kernel_size=(5,5), stride=(1, 1))

        
    def forward(self, x):
        
        # encoder
        x = F.relu(self.enc_conv0(x))
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = self.enc_conv3(x)

        # decoder
        x = F.relu(self.dec_conv0(x))
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv3(x)

        return x
