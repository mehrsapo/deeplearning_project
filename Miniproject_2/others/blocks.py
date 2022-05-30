from torch import empty
from torch.nn.functional import fold, unfold


class Module (object):
    """
    parent class for modules
    """
    def forward (self, *input) :
        raise NotImplementedError

    def backward (self, *gradwrtoutput) :
        raise NotImplementedError

    def param(self) :
        return []


class Conv2d(Module) : 

    def __call__(self , input):
        return self.forward(input)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation =1, bias=True, padding_mode='zeros', device='cpu', dtype=None):
    
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(stride, int):
            stride = (stride, stride)

        self.stride = stride
        
        
        if isinstance(padding, int):
            padding = (padding, padding)

        self.padding = padding
        
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.dilation = dilation

        self.device = device

        k = 1/(in_channels * kernel_size[0] * kernel_size[1])
        
        ### Initializing weights from uniform [-k, k] distribution (as in Pytorch)
        self.weight = empty(out_channels, in_channels, kernel_size[0], kernel_size[1]).uniform_(- k ** 0.5, k** 0.5).to(device)
        self.bias = empty(out_channels).uniform_(- k ** 0.5, k** 0.5).to(device)
        
        ### Initializing gradients by zero
        self.weight.grad = empty(out_channels, in_channels, kernel_size[0], kernel_size[1]).zero_().to(device)
        self.bias.grad = empty(out_channels).zero_().to(device)

        self.if_bias = bias
        
        if(not bias):
            self.bias.fill_(0)


    def forward(self, input):
    
        ### Saving the input for the backward pass
        self.input = input
        
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]
        
        s_h = self.stride[0]
        s_w = self.stride[1]
        
        p_h = self.padding[0]
        p_w = self.padding[1]
        
        d_h = self.dilation[0]
        d_w = self.dilation[1]
        
        ### computing the effective kernle due to the dilation
        dialeted_kernerl_len_h = k_h + (k_h-1)*(d_h-1)
        dialeted_kernerl_len_w = k_w + (k_w-1)*(d_w-1)
        
        h_in, w_in = input.shape[2], input.shape[3]
        
        self.h_in = h_in
        self.w_in = w_in
        
        ### preparing the input matrix for multiplication as the second matrix in eq. 1 in the report
        input_unfold = unfold(input, (dialeted_kernerl_len_h, dialeted_kernerl_len_w), padding=(p_h, p_w), stride=(s_h, s_w))
        
        ### adding zeros to the kernel if we have dilation
        if d_h!=1 or d_w!=1 :
            W_a = empty(self.weight.size(0), self.weight.size(1), dialeted_kernerl_len_h, dialeted_kernerl_len_w).fill_(0).to(self.device)
            W_a[:,:,::d_h, ::d_w] = self.weight
        else:
            W_a = self.weight
        
        self.W_a = W_a

        out = W_a.view(W_a.size(0), -1).matmul(input_unfold)+ self.bias.view(1, -1, 1)

        h_out = int( (h_in + 2 * p_h - d_h*(k_h - 1) - 1) / s_h + 1 )
        w_out = int( (w_in + 2 * p_w - d_w*(k_w - 1) - 1) / s_w + 1 )
        
        ### matricizing the output
        out = fold(out, (h_out, w_out),  (1,1))
                    
        return out


    def W_grad(self, gradwrtoutput):
    
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]
        
        s_h = self.stride[0]
        s_w = self.stride[1]
        
        p_h = self.padding[0]
        p_w = self.padding[1]
        
        d_h = self.dilation[0]
        d_w = self.dilation[1]
        
        k_h_e = k_h + (k_h-1)*(d_h-1)
        k_w_e = k_w + (k_w-1)*(d_w-1)
        
        k1 = gradwrtoutput.size(2)
        k2 = gradwrtoutput.size(3)
        
        x = self.input
        
        F = gradwrtoutput.transpose(0,1)
        
        
        ### padding/ dilating the gradient matrix if we have stride > 1
        if s_h!=1 or s_w!=1 :
        
            ### for matching the size
            a =  x.size(2) + 2*p_h - k_h_e - k1 - (k1-1) * (s_h-1) +1
            b = x.size(3) + 2*p_w - k_w_e - k2 - (k2-1) * (s_w-1) +1


            k1_n = k1 + (k1-1) * (s_h-1) + a
            k2_n = k2 + (k2-1) * (s_w-1) + b

            T = empty(F.size(0), F.size(1), k1_n, k2_n).fill_(0).to(self.device)
            T[:,:,::s_h, ::s_w] = F
            
            F = T
            
            k1 = T.size(2)
            k2 = T.size(3)

        ### preparing the input matrix for multiplication
        x_unfold = unfold(x.transpose(0, 1), (k1, k2), padding=(p_h, p_w), stride=1)
        
        out = F.reshape(F.size(0), -1).matmul(x_unfold)

        out = out.transpose(0,1)

        ### matricizing the output
        out = fold(out, (k_h_e, k_w_e), (1, 1))
        
        ### extracting the true gradient if we have dilation
        if d_h!=1 or d_w!=1 :
            out = out[:,:,::d_h,::d_w]
        
        return out


    def B_grad(self, gradwrtoutput):
        return gradwrtoutput.sum((0,2,3)).view_as(self.bias)


    def backward(self, gradwrtoutput): 

        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]
        
        s_h = self.stride[0]
        s_w = self.stride[1]
        
        p_h = self.padding[0]
        p_w = self.padding[1]
        
        d_h = self.dilation[0]
        d_w = self.dilation[1]
        
        k_h_e = k_h + (k_h-1)*(d_h-1)
        k_w_e = k_w + (k_w-1)*(d_w-1)

        if(self.if_bias):
            self.bias.grad = self.B_grad(gradwrtoutput)

        self.weight.grad = self.W_grad(gradwrtoutput)
        
        ### rotating the (dilated) kernel matrix
        W_a = self.W_a
        W_a = W_a.view(W_a.size(0), W_a.size(1), -1)
        W_a = W_a[:,:,range(W_a.size(2)-1,-1,-1)]
        W_a = W_a.view(W_a.size(0), W_a.size(1), k_h_e,k_w_e)
        
        W_a = W_a.transpose(0,1)
        
        x = gradwrtoutput
        
        k1 = x.size(2)
        k2 = x.size(3)
        
        k1_n = k1 + (k1-1) * (s_h-1)
        k2_n = k2 + (k2-1) * (s_w-1)
        
        ### dilating the gradient matrix if we have stride >1
        if s_h!=1 or s_w!=1 :

            T = empty(x.size(0), x.size(1), k1_n, k2_n).fill_(0).to(self.device)
            T[:,:,::s_h, ::s_w] = x
            
            x = T

        ### padding to match the size, number of padding is equal to the number entries which are not involved in the original convolution
        a = (self.h_in + 2*p_h - k_h_e)%s_h   ###### leftovers
        b = (self.w_in + 2*p_w - k_w_e )%s_w
        
        A = empty(x.size(0), x.size(1), k1_n+a, k2_n+b).fill_(0).to(self.device)
        
        if a > 0 and b>0:
            A[:,:,:-a,:-b] = x
            x = A
        elif a>0:
            A[:,:,:-a,:] = x
            x = A
        elif b>0:
            A[:,:,:,:-b] = x
            x = A

        a =  p_h + (k_h_e -1)/2 + (self.h_in - (k1_n+a))/2
        b =  p_w + (k_w_e -1)/2 + (self.w_in - (k2_n+b))/2
        
        ### preparing the input matrix for multiplication with the proper padding computed above
        x_unfold = unfold(x, (k_h_e, k_w_e), padding=(int(a),int(b)), stride=1)

        out = W_a.reshape(W_a.size(0), -1).matmul(x_unfold)
        
        ### matricizing the output
        out = fold(out, (self.h_in+ 2*p_h, self.w_in+2*p_w), (1, 1))
        
        ### extracting the true gradient of input if the input has been padded
        if p_h!=0 or p_w!=0 :
            out = out[:,:,p_h:-p_h, p_w:-p_w]

        return out


    def param(self):
        if self.if_bias:
            return [(self.weight, self.weight.grad), (self.bias, self.bias.grad)]
        else: 
            return [(self.weight, self.weight.grad)]
    


class TransposeConv2d(Module):

    def __call__(self , input):
        return self.forward(input)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation =1, bias=True, padding_mode='zeros', device='cpu', dtype=None):
    
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(stride, int):
            stride = (stride, stride)

        self.stride = stride
        
        
        if isinstance(padding, int):
            padding = (padding, padding)

        self.padding = padding
        
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.dilation = dilation
        
        self.device = device
        
        k = 1/(out_channels * kernel_size[0] * kernel_size[0])

        ### Initializing weights from uniform [-k, k] distribution (as in Pytorch)
        self.weight = empty(in_channels, out_channels, kernel_size[0], kernel_size[1]).uniform_(- k ** 0.5, k** 0.5).to(device)
        self.bias = empty(out_channels).uniform_(- k ** 0.5, k** 0.5).to(device)
        
        ### Initializing gradients by zero
        self.weight.grad = empty(in_channels, out_channels, kernel_size[0], kernel_size[1]).zero_().to(device)
        self.bias.grad = empty(out_channels).zero_().to(device)

        self.if_bias = bias
        if(not bias):
            self.bias.fill_(0)


    def forward(self,x):

        ### Saving the input for the backward pass
        self.input = x
                
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]
        
        s_h = self.stride[0]
        s_w = self.stride[1]
        
        p_h = self.padding[0]
        p_w = self.padding[1]
        
        d_h = self.dilation[0]
        d_w = self.dilation[1]

        ### computing the effective kernle due to the dilation
        dialeted_kernerl_len_h = k_h + (k_h-1)*(d_h-1)
        dialeted_kernerl_len_w = k_w + (k_w-1)*(d_w-1)
        
        h_in, w_in = x.shape[2], x.shape[3]
        
        self.h_in = h_in
        self.w_in = w_in
        
        
        k1_n = h_in + (h_in-1) * (s_h-1)
        k2_n = w_in + (w_in-1) * (s_w-1)
            
        ### adding zeros to the input if we have stride>1
        if s_h!=1 or s_w!=1 :
            T = empty(x.size(0), x.size(1), k1_n, k2_n).fill_(0).to(self.device)
            T[:,:,::s_h, ::s_w] = x
            
            x = T

         ### preparing the input matrix for multiplication
        x_unfold = unfold(x, (dialeted_kernerl_len_h, dialeted_kernerl_len_w), padding=(dialeted_kernerl_len_h-1-p_h, dialeted_kernerl_len_w-1-p_w) , stride=1)

        ### adding zeros to the kernel if we have dilation
        if d_h!=1 or d_w!=1 :
            W_a = empty(self.weight.size(0), self.weight.size(1), dialeted_kernerl_len_h, dialeted_kernerl_len_w).fill_(0).to(self.device)
            W_a[:,:,::d_h, ::d_w] = self.weight
        else:
            W_a = self.weight
            
        self.W_a = W_a
        
        W_a = W_a.transpose(0,1)
        
        ### rotating the kernel
        W_a = W_a.view(W_a.size(0), W_a.size(1), -1)
        W_a = W_a[:,:,range(W_a.size(2)-1,-1,-1)]
        W_a = W_a.view(W_a.size(0), W_a.size(1), dialeted_kernerl_len_h,dialeted_kernerl_len_w)
        

        out = W_a.reshape(W_a.size(0), -1).matmul(x_unfold) + self.bias.view(1, -1, 1)
        
        h_out = (h_in - 1)*s_h - 2*p_h + d_h*(k_h-1) + 1
        w_out = (w_in - 1)*s_w - 2*p_w + d_w*(k_w-1) + 1
        
        ### matricizing the output
        out = fold(out, (h_out, w_out),  (1,1))
                
        return out
        
        

    def W_grad(self, gradwrtoutput):
    
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]
        
        s_h = self.stride[0]
        s_w = self.stride[1]
        
        p_h = self.padding[0]
        p_w = self.padding[1]
        
        d_h = self.dilation[0]
        d_w = self.dilation[1]
        
        h_in = self.h_in
        w_in = self.w_in
        
        dialeted_kernerl_len_h = k_h + (k_h-1)*(d_h-1)
        dialeted_kernerl_len_w = k_w + (k_w-1)*(d_w-1)
        
        k1 = gradwrtoutput.size(2)
        k2 = gradwrtoutput.size(3)
        
        x = self.input
        
        F = gradwrtoutput.transpose(0,1)
            
        k1_n = h_in + (h_in-1) * (s_h-1)
        k2_n = w_in + (w_in-1) * (s_w-1)
        
        ### dilating the input matrix if we have stride > 1
        if s_h!=1 or s_w!=1 :


            T = empty(x.size(0), x.size(1), k1_n, k2_n).fill_(0).to(self.device)
            T[:,:,::s_h, ::s_w] = x
            
            x = T

        ### preparing the input matrix for multiplication
        x_unfold = unfold(x.transpose(0, 1), (k1, k2), padding=(dialeted_kernerl_len_h-1-p_h, dialeted_kernerl_len_w-1-p_w), stride=1)
        
        out = F.reshape(F.size(0), -1).matmul(x_unfold)

        ### matricizing the output
        out = fold(out, (dialeted_kernerl_len_h, dialeted_kernerl_len_w), (1, 1))
        
        
        ### rotate to get the gradient wrt true kernel
        out = out.reshape(out.size(0), out.size(1), -1)
        out = out[:,:,range(out.size(2)-1,-1,-1)]
        out = out.reshape(out.size(0), out.size(1), dialeted_kernerl_len_h,dialeted_kernerl_len_w)
        
        ### extracting the true gradient if we have dilation
        if d_h!=1 or d_w!=1 :
            out = out[:,:,::d_h,::d_w]
        
        return out

    def B_grad(self, gradwrtoutput):
        return gradwrtoutput.sum((0,2,3)).view_as(self.bias)


    def backward(self, gradwrtoutput):
    
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]
        
        s_h = self.stride[0]
        s_w = self.stride[1]
        
        p_h = self.padding[0]
        p_w = self.padding[1]
        
        d_h = self.dilation[0]
        d_w = self.dilation[1]
        
        h_in = self.h_in
        w_in = self.w_in
        
        dialeted_kernerl_len_h = k_h + (k_h-1)*(d_h-1)
        dialeted_kernerl_len_w = k_w + (k_w-1)*(d_w-1)

        if(self.if_bias):
            self.bias.grad = self.B_grad(gradwrtoutput)

        self.weight.grad = self.W_grad(gradwrtoutput)
                
        W_a = self.W_a
        
        x = gradwrtoutput

        ### preparing the input matrix of convolution for multiplication with the proper padding computed above
        x_unfold = unfold(x, (dialeted_kernerl_len_h, dialeted_kernerl_len_w), padding=(dialeted_kernerl_len_h-1,dialeted_kernerl_len_w-1), stride=1)

        out = W_a.reshape(W_a.size(0), -1).matmul(x_unfold)

        
        h_out_raw = h_in + (h_in-1) * (s_h-1)+ 2*((dialeted_kernerl_len_h-1)-p_h)
        w_out_raw = w_in + (w_in-1) * (s_w-1)+2*((dialeted_kernerl_len_w-1)-p_w)

        ### matricizing the output
        out = fold(out, (h_out_raw, w_out_raw), (1, 1))
        
        added_pad_h = (dialeted_kernerl_len_h-1)-p_h
        added_pad_w = (dialeted_kernerl_len_w-1)-p_w
        
        ### extracting the true gradient of input
        out = out[:,:,added_pad_h:-added_pad_h, added_pad_w:-added_pad_w]
        
        
        ### extracting the true gradient of input if zeros columns/rows have been inserted due to the stride >1
        if s_h!=1 or s_w!=1 :
            out = out[:,:,::s_h,::s_w]

        return out


    def param(self):
        if self.if_bias:
            return [(self.weight, self.weight.grad), (self.bias, self.bias.grad)]
        else: 
            return [(self.weight, self.weight.grad)]
        



class Upsampling(TransposeConv2d):
        """
        Upsampling is a wrapper around TransposeConv2d module
        """
        def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, stride=1, padding_mode='zeros', device='cpu', dtype=None):
            super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, device=device, dtype=dtype)

        def forward(self, x):
            return super().forward(x)

        def B_grad(self, gradwrtoutput):
            return super().B_grad(gradwrtoutput)

        def W_grad(self, gradwrtoutput):
            return super().W_grad(gradwrtoutput)

        def backward(self, gradwrtoutput):
            return super().backward(gradwrtoutput)

        def param(self):
            return super().param()


class ReLU(Module):

    def __call__(self , input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        return (input > 0).float().mul(input) 

    def backward(self, gradwrtoutput):
        return (self.input > 0).float().mul(gradwrtoutput)

    def param(self):
        return []


class Sigmoid(Module):

    def __call__(self , input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        return 1 / (1 + (-input).exp())
    
    def backward(self, gradwrtoutput):
        return ((-self.input).exp().div((1 + (-self.input).exp()) ** 2)).mul(gradwrtoutput)

    def param(self):
        return []


class Sequential(Module): 
    '''
    This module combines given modules in a sequntial order
    '''
    def __call__(self , input):
        return self.forward(input)

    def __init__(self, *args) -> None:
        self.layers = args

    
    def forward(self, input):
        for layer in self.layers: 
            input = layer.forward(input)
        
        return input


    def backward(self, gradwrtoutput):
        for layer in reversed(self.layers): 
            gradwrtoutput = layer.backward(gradwrtoutput)

        return gradwrtoutput

    def param(self):
        params = list()

        for layer in self.layers:
            params.extend(layer.param())

        return params

    def update_parameters(self, parameters):
        """
        This function updates the parameters to keep track of changes during optimizing 
        """
        counter = -1

        for layer in self.layers: 
            if isinstance(layer, Conv2d) or isinstance(layer, TransposeConv2d):
                counter += 1
                layer.weight = parameters[counter][0]
                layer.weight.grad = parameters[counter][1]
                if layer.if_bias:
                    counter += 1
                    layer.bias = parameters[counter][0]
                    layer.bias.grad = parameters[counter][1]



class MSE(Module):
    """
    MSE loss module
    """
    def __call__(self, output, target):
        return self.forward(output, target)

    def forward(self, output, target):
        self.output = output
        self.target = target
        return (output - target).pow(2).mean()

    def backward(self):
        return (2*(self.output - self.target)) / self.output.numel()

    def param(self):
        return []


class SGD(Module):
    """
    optimizer: Stochastic Gradient descent 
    """
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        """
        each step does a gradient descent on the input batch
        """
        for (var, grad_var) in self.params: 
            var.sub_(grad_var.mul_(self.lr))
    
    def zero_grad(self):
        for (_, grad_var) in self.params:
            grad_var.zero_()

    def update_parameters(self, params):
        self.params = params

    def print_param(self):
        print(self.params)