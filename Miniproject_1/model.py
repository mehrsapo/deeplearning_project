import torch 
from . networks.unet import UNet 

class Model():

    def __init__(self) -> None: 
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        self.net = UNet()
        self.net = self.net.to(self.device)

        


        pass

    def load_pretrained_model(self) -> None: 

        pass

    def train(self, train_input, train_target, num_epochs) -> None: 
        
        pass

    def predict(self, test_input) -> torch.Tensor: 

        pass

    