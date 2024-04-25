import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange,repeat

#################################################################################
#                                   IMAGE BASED                                 #
#################################################################################


def image_mse_loss_mask(predict: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
    predict : (N,C,H,W)
    target : (N,C,H,W)
    mask : (H,W) or (N,H,W)  1 -> mask,0 -> unmask

    Return:
    loss : float
    """
    N,C,H,W = predict.shape

    # mask
    if mask is None:
        return F.mse_loss(predict, target, reduction='mean')
    elif mask.dim() == 2: # (H,W)
        index = mask == 0
        index = repeat(index,'h w -> n c h w',n=N,c=C)
        return F.mse_loss(predict[index], target[index], reduction='mean')
    elif mask.dim() == 3: # (N,H,W)
        index = mask == 0
        index = repeat(index,'n h w -> n c h w',c=C)
        return F.mse_loss(predict[index], target[index], reduction='mean')



#################################################################################
#                                   VIDEO BASED                                 #
#################################################################################

def video_mse_loss_mask(predict: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
    predict : (N,T,C,H,W)
    target : (N,T,C,H,W)
    mask : (H,W) or (T,H,W) or (N,H,W) or (N,T,H,W)  1 -> mask,0 -> unmask

    Return:
    loss : float
    """
    N,T,C,H,W = predict.shape

    if mask is None:
        mask = torch.ones(H,W).to(predict.device)
    else:
        index = mask == 0
        if tuple(mask.shape) == (H,W):
            index = repeat(index,'h w -> n t c h w',n=N,t=T,c=C)
            return F.mse_loss(predict[index], target[index], reduction='mean')
        elif tuple(mask.shape) == (T,H,W):
            index = repeat(index,'t h w -> n t c h w',n=N,c=C)
            return F.mse_loss(predict[index], target[index], reduction='mean')
        elif tuple(mask.shape) == (N,H,W):
            index = repeat(index,'n h w -> n t c h w',t=T,c=C)
            return F.mse_loss(predict[index], target[index], reduction='mean')
        elif tuple(mask.shape) == (N,T,H,W):
            index = repeat(index,'n t h w -> n t c h w',c=C)
            return F.mse_loss(predict[index], target[index], reduction='mean')
        else:
            raise ValueError("mask shape is not supported")
