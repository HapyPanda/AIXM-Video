import torch
import sys
sys.path.append(".")
import numpy as np
import random
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict

from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl

from aixmvideo.models import VDT
from aixmvideo.dataset import VideoDataset



# Args
@dataclass
class TrainingArguments:
    # Train
    exp_name: str       = field(default="daydreamer",metadata={"help": "The name of the experiment."})
    batch_size: int     = field(default=1, metadata={"help": "The number of samples per training iteration."})
    precision: str      = field(default="bf16",metadata={"help": "The precision type used for training."})
    output_dir: str     = field(default="results/daydreamer",metadata={"help": "The directory where training results are saved."})
    save_steps: int     = field(default=2000,metadata={"help": "The interval at which to save the model during training."})
    max_steps: int      = field(default=100000,metadata={"help": "The maximum number of steps for the training process."})
    n_nodes: int        = field(default=1, metadata={"help": "The number of nodes used for training."})
    devices: int        = field(default=8, metadata={"help": "The number of devices used for training."})
    num_workers: int    = field(default=8,metadata={"help": "The number of subprocesses used for data handling."},)
    resume_from_checkpoint: str = field(default=None, metadata={"help": "Resume training from a specified checkpoint."})
    load_from_checkpoint: str = field(default=None, metadata={"help": "Load the model from a specified checkpoint."})

    # Data
    video_path: str     = field(default="/remote-home1/dataset/data_split_tt",metadata={"help": "The path where the video data is stored."})
    video_num_frames:int= field(default=17,metadata={"help": "The number of frames per video."})
    resolution: int     = field(default=256, metadata={"help": "The resolution of the videos."})
    sample_rate: int    = field(default=1,metadata={"help": "The sampling interval."})
    dynamic_sample: bool = field(default=False, metadata={"help": "Whether to use dynamic sampling."})

    # Model
    model_config: str   = field(default="configs/causalvae.yaml",metadata={"help": "The path to the model configuration file."})

# seed everything
def set_seed(seed=1006):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Callbacks in Pytorch Lightning
def load_callbacks(ars):
    checkpoint_callback = ModelCheckpoint(
        dirpath = args.output_dir,
        filename="model-{epoch:02d}-{step}",
        every_n_train_steps=args.save_steps,
        save_top_k=5,
        save_on_train_epoch_end=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [checkpoint_callback, lr_monitor]

def train(args):
    set_seed()
    # Load Config
    model = VDT()
    if args.load_from_checkpoint is not None:
        model = CausalVAEModel.from_pretrained(args.load_from_checkpoint)
    else:
        model = CausalVAEModel.from_config(args.model_config)

    if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
        print(model)

    # Load Dataset
    dataset = VideoDataset(
        args.video_path,
        sequence_length=args.video_num_frames,
        resolution=args.resolution,
        sample_rate=args.sample_rate,
        dynamic_sample=args.dynamic_sample,
    )
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True,
    )
    
    # Load Callbacks and Logger
    callbacks = load_callbacks(args)

    # Load Trainer
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=args.devices,
        num_nodes=args.n_nodes,
        callbacks=callbacks,
        log_every_n_steps=5,
        precision=args.precision,
        max_steps=args.max_steps,
        strategy="ddp_find_unused_parameters_true",
    )
    trainer_kwargs = {}
    if args.resume_from_checkpoint:
        trainer_kwargs["ckpt_path"] = args.resume_from_checkpoint

    trainer.fit(model, train_loader, **trainer_kwargs)
    # Save Huggingface Model
    model.save_pretrained(os.path.join(args.output_dir, "hf"))



if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    args = parser.parse_args_into_dataclasses()[0] # multi args,choose the first one