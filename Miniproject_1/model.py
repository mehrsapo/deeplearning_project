import torch 
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from . networks.simple_unet import SUet 
from . utils.metrics import psnr


class Model():

    def __init__(self) -> None: 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.batch_size = 32

        self.net = SUet()
        self.net = self.net.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100,150], gamma=0.1)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9)

        pass

    def load_pretrained_model(self) -> None: 

        pass

    def train(self, train_input, train_target, valid_in, valid_out, num_epochs) -> None: 

        train_input = train_input.float().to(self.device) / 255.0
        train_target = train_target.float().to(self.device) / 255.0
        valid_in = valid_in.float().to(self.device) / 255.0
        valid_out = valid_out.float().to(self.device) / 255.0

        print(train_input.max(), train_input.min())
        
        train_dataset = data.TensorDataset(train_input, train_target)

        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs): 

            running_loss = 0.0
            for i, (inputs, targests) in enumerate(train_loader, 0): 

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targests)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 2000 == 0:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000}')
                    print(psnr(self.predict(valid_in), valid_out))
                    running_loss = 0.0

        pass

    def predict(self, test_input) -> torch.Tensor: 
        return self.net(test_input)


    