import torch
import ipdb
from diffusers.schedulers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

shape = (2,3,4,8,8)
a = torch.randn(shape)
noise = torch.randn(shape)


scheduler = DDPMScheduler(num_train_timesteps=1000)
ipdb.set_trace()
sample = scheduler.add_noise(a, noise, t=99)