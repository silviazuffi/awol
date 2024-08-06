import torch

def mse_loss(x, x_gt, type='L1'):
    if type == 'L1':
        criterion = torch.nn.L1Loss()
        loss =  criterion(x, x_gt)
    elif type == 'L2':
        loss = torch.sum(x - x_gt)**2.
    return loss
