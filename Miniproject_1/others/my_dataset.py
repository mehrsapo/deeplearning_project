import torch.utils.data as data
from torchvision import transforms
import random

class My_Dataset(data.Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __getitem__(self, index):
        input = self.input[index]
        output = self.output[index]

        if random.random() < 0.5:
            input = transforms.functional.hflip(input)
            output = transforms.functional.hflip(output)

        if random.random() < 0.5:
            input = transforms.functional.vflip(input)
            output = transforms.functional.vflip(output)

        '''if random.random() < 0.2:
            angle = random.random() * 180
            input = transforms.functional.rotate(input, angle)
            output = transforms.functional.rotate(output, angle)
            
        if random.random() < 0.4:
            output = transforms.functional.gaussian_blur(output, 3, 0.1)    
        '''

        if random.random() < 0.2:
            br = random.randint(2, 10)
            input = transforms.functional.adjust_brightness(input, br)
            output = transforms.functional.adjust_brightness(output, br)

            
        return input, output

    def __len__(self):
        return self.input.size(0)