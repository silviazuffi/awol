### Code modified from https://github.com/AutodeskAILab/Clip-Forge/blob/main/networks/latent_flows.py

####### Code build on top of  https://github.com/ikostrikov/pytorch-flows


import math
import types

import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-6 
from collections import OrderedDict



class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var + EPS)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var + EPS)).sum(
                -1, keepdim=True)

        
class CouplingLayer(nn.Module):

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu',
                 train_mask=False,
                 invert=False,
                 no_compression=False):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask
        self.train_mask = train_mask
        self.invert = invert

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs
      
        if no_compression:
            # Use this for "no_comp"
            self.scale_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), s_act_func(),
                nn.Linear(num_hidden, num_hidden), s_act_func(),
                nn.Linear(num_hidden, num_inputs))

        
            self.translate_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), t_act_func(),
                nn.Linear(num_hidden, num_hidden), t_act_func(),
                nn.Linear(num_hidden, num_inputs))
        else:
        

            self.scale_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), s_act_func(),
                nn.Linear(num_hidden, 512), s_act_func(),
                nn.Linear(512, num_hidden), s_act_func(),
                nn.Linear(num_hidden, num_inputs))

        
            self.translate_net = nn.Sequential(
                nn.Linear(total_inputs, num_hidden), t_act_func(),
                nn.Linear(num_hidden, 512), t_act_func(),
                nn.Linear(512, num_hidden), t_act_func(),
                nn.Linear(num_hidden, num_inputs))
        

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        '''
        The out variable was not present in this code, but I think it should be there
        '''
        mask = self.mask
        if self.train_mask:
            masked_inputs = mask.forward(inputs, self.invert, cond_inputs)
            inv_masked_input = mask.forward(inputs, ~self.invert, cond_inputs)
        else:
            masked_inputs = inputs * mask
            inv_masked_input = inputs * torch.abs(mask-1)

        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if self.train_mask:
            if mode == 'direct':
                log_s = self.scale_net(masked_inputs)  
                t = self.translate_net(masked_inputs) 
                s = torch.exp(log_s)
                return inv_masked_input + inputs * s + t, log_s.sum(-1, keepdim=True)
            else:
                log_s = self.scale_net(masked_inputs) 
                t = self.translate_net(masked_inputs) 
                s = torch.exp(-log_s)
                return inv_masked_input + (inputs - t) * s, -log_s.sum(-1, keepdim=True)
        else:
            if mode == 'direct':
                log_s = self.scale_net(masked_inputs) * (1 - mask)
                t = self.translate_net(masked_inputs) * (1 - mask)
                s = torch.exp(log_s)
                return inputs * s + t, log_s.sum(-1, keepdim=True)
            else:
                log_s = self.scale_net(masked_inputs) * (1 - mask)
                t = self.translate_net(masked_inputs) * (1 - mask)
                s = torch.exp(-log_s)
                return (inputs - t) * s, -log_s.sum(-1, keepdim=True)



class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']

        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_prob(self, inputs, cond_inputs = None):
        u, log_jacob = self.forward(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi + EPS)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None, num_inputs=None, sigma=1.0):
        if noise is None:
            noise = sigma*torch.Tensor(num_samples, num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples
   
class MaskGenerator(nn.Module):
   def __init__(self, num_inputs, num_cond=0):
       super(MaskGenerator, self).__init__()

       # If the mask is generated conditioned on some variables, or just trained as a parameter
       if num_cond==0:
           self.mask_cond = False
           self.mask = nn.Parameter(2*torch.rand(num_inputs)-1, requires_grad=True)
       else:
           self.mask_cond = True
           self.pred_layer = nn.Linear(num_cond, num_inputs)

   def forward(self, x, invert=False, cond=None):
       if self.mask_cond:
           self.mask = self.pred_layer(cond)
       # Apply a sigmoid function to the parameter to get values between 0 and 1
       mask_values = torch.sigmoid(self.mask)
       # Apply a threshold to convert values to 0 or 1
       binary_mask = torch.round(mask_values)
       masked_output = x * binary_mask
       return masked_output
    
def get_generator(num_inputs, num_cond_inputs, device, flow_type="realnvp", num_blocks=5, num_hidden=1024, train_mask=False, mask_conditioning=False, no_compression=False):
    num_blocks = num_blocks
    modules = []
    num_inputs = num_inputs
    device =device
    num_cond_inputs = num_cond_inputs
    num_hidden = num_hidden
    flow_type = flow_type
    if flow_type== 'realnvp': ### Checkered Masking
        if train_mask:
            masks = [] #MaskGenerator(num_inputs)
        else:
            mask = torch.arange(0, num_inputs) % 2
            mask = mask.to(device).float()

        invert = False
        for i in range(num_blocks):
            # Generate the mask for the even blocks, and use the invert for the odd blocks
            if train_mask:
                if True: #~(i % 2):
                    if mask_conditioning:
                        masks += [MaskGenerator(num_inputs, num_cond_inputs)]
                    else:
                        masks += [MaskGenerator(num_inputs)]
                    mask = masks[i]
                    invert = False
                else:
                    invert = True

            modules += [
                CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu', train_mask=train_mask, invert=invert, no_compression=no_compression), #~invert),
                BatchNormFlow(num_inputs)
            ]

            if not train_mask:
                mask = 1 - mask

        generator = FlowSequential(*modules).to(device)
    
    elif flow_type == 'realnvp_half': ### Dimesion Masking
        mask = (torch.arange(0, num_inputs) < (num_inputs/2)).type(torch.uint8)
        mask = mask.to(device).float()
        for _ in range(num_blocks):
            modules += [
                CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu', no_compression=no_compression),
                BatchNormFlow(num_inputs)
            ]
            mask = 1 - mask        
        generator = FlowSequential(*modules).to(device) 

    return generator    
