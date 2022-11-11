import torch

def mse_loss(output, target):
    loss_fn = torch.nn.MSELoss()
    
    return loss_fn(output, target)