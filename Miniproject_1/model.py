from random import shuffle
import torch 
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from . networks.unet import UNet 

class Model():

    def __init__(self) -> None: 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4

        self.net = UNet()
        self.net = self.net.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimzer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8)




        pass

    def load_pretrained_model(self) -> None: 

        pass

    def train(self, train_input, train_target, num_epochs) -> None: 
        
        train_dataset = data.TensorDataset(train_input.float().to(self.device), train_target.float().to(self.device))

        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs): 

            running_loss = 0.0
            for i, train_data in enumerate(train_loader): 
                
                inputs, targests = train_data

                self.optimzer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targests)
                loss.backward()
                self.optimzer.step()
                
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0



        pass

    def predict(self, test_input) -> torch.Tensor: 

        pass

    