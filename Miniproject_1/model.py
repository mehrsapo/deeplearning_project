import os
import torch 
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from . others.unet import UNet
# from . others.ten_conv_net import TenConv
from . others.my_dataset import My_Dataset


class Model():

    def __init__(self) -> None: 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.batch_size = 32

        self.net = UNet() #TenConv()
        self.net = self.net.to(self.device)

        self.criterion = nn.MSELoss()   # chosen loss function 
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8)   # chosen optimizer
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60], gamma=0.1)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)            
        # self.optimizer = optim.Adagrad(self.net.parameters(), lr=0.001)

        pass

    def load_pretrained_model(self) -> None: 
        self.net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'bestmodel.pth')))
        pass

    def train(self, train_input, train_target, num_epochs) -> None: 

        train_input = train_input.float().to(self.device) / 255.0
        train_target = train_target.float().to(self.device) / 255.0

        train_dataset = My_Dataset(train_input, train_target)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        batch_per_epoch = int(train_input.size(0)/self.batch_size)-1

        # train loop: 
        
        for epoch in range(num_epochs): 

            running_loss = 0.0
            for i, (inputs, targests) in enumerate(train_loader, 0): 

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targests)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i == batch_per_epoch: 
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_per_epoch}')
                    running_loss = 0.0
            
            self.scheduler.step()

        pass



    def predict(self, test_input) -> torch.Tensor: 
        return (self.net(test_input.float().to(self.device) / 255.0) * 255).cpu()


    