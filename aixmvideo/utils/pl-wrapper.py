import torch
import torch.nn as nn
import pytorch_lightning as pl


#############################################
#      Pytorch Lightning for Diffusion      #
#############################################

class PL_Diffusion(pl.LightningModule):
    
    def __init__(self, model, config):
        super().__init__()
        self.model = model  
        self.config = config

    def register_diffusion_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def training_step(self,batch,batch_idx) :
        """
        Args:

        """
        pass