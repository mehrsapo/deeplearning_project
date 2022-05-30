from . others.blocks import *
import torch.utils.data as data
import torch   # only for torch.device and torch.set_grad_enabled(False) and torch.cuda.is_available()
import pickle

import os 


class Model():

    def __init__(self) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32

        self.net = Sequential(Conv2d(3, 64, 2,  stride = 2, device=self.device),
                              ReLU(), 
                              Conv2d(64, 256, 2,  stride = 2, device=self.device),
                              ReLU(),
                              Upsampling(256, 64, 2,  stride = 2, device=self.device), 
                              ReLU(),
                              Upsampling(64, 3, 2,  stride = 2, device=self.device), 
                              Sigmoid())

        self.optimizer =  SGD(self.net.param(), 10)
        self.criterion = MSE()


        pass


    def load_pretrained_model(self) -> None: 
        with open(os.path.join(os.path.dirname(__file__), 'bestmodel.pth'), 'rb') as handle:
            best_model = pickle.load(handle)

        params = best_model['params']           # best model are saved as a dictionary with best_model['params'] being the parameters
        params_correct_device = list()

        # handling right device for parameter:
        for (var, grad_var) in params: 
            params_correct_device.append((var.to(device = self.device), grad_var.to(device = self.device)))

        # update parameters in both network and optimizer
        self.net.update_parameters(params_correct_device)
        self.optimizer.update_parameters(params_correct_device)

        pass


    def train(self, train_input, train_target, num_epochs) -> None: 
     
        torch.set_grad_enabled(False)

        train_input = train_input.float().to(self.device) / 255.0
        train_target = train_target.float().to(self.device) / 255.0


        train_dataset = data.TensorDataset(train_input, train_target)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        batch_per_epoch = int(train_input.size(0)/self.batch_size)-1
        
        # train loop: 

        for epoch in range(num_epochs): 
   
            running_loss = 0.0
    
            for i, (inputs, targets) in enumerate(train_loader, 0): 
                # when a parmeter is updated, it should be both updated in net and the optimizer
                self.optimizer.zero_grad()
                self.net.update_parameters(self.optimizer.params)
                outputs = self.net(inputs)
                loss_my = self.criterion(outputs, targets)
                self.net.backward(self.criterion.backward())
                self.optimizer.update_parameters(self.net.param())
                self.optimizer.step()
                self.net.update_parameters(self.optimizer.params)
                
                running_loss += loss_my.item() 

                if i == batch_per_epoch: 
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_per_epoch}')
                    running_loss = 0.0

        pass


    def predict(self, test_input) -> empty: 
        return (self.net(test_input.float().to(self.device) / 255.0) * 255).cpu()