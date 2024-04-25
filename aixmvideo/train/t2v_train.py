import sys
# sys.path.append("/mnt/lpai-dione/ssai/cvg/team/didonglin/lhz/AIXM-Video/aixmvideo")

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
print(sys.path)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
# from .dataset import getdataset
from dataset import getdataset
import argparse
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import imageio

def main(args):
    device = torch.device("cuda")
    dtype = torch.float32
    vae = AutoencoderKL.from_pretrained("/mnt/lpai-dione/ssai/cvg/team/didonglin/lhz/Latte/pretrained_model/t2v_required_models/vae").to(device,dtype)
    vae.requires_grad_(False)

    train_dataset = getdataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=Collate(args),  # TODO: do not enable dynamic mask in this point
        batch_size=args.train_batch_size,
        # num_workers=args.dataloader_num_workers,
    )
    i = 1
    for data in train_dataloader:
        if args.dataset == "t2v":
            video,idx,mask = data
            video = video.to(device)
            print(video.shape)
            x = rearrange(video,'b c f h w -> (b f) c h w').contiguous()
            print(x.shape)
            # sample 从目标分布中采样得到一个样本
            z = vae.encode(x).latent_dist.sample().mul(0.18215)
            print(z.shape)
            # zframe1 = z[:1,:1,:,:]
            # zframe2 = z[16:17,:1,:,:]
            # zframe = zframe.squeeze(0)
            # save_image(zframe1,'/mnt/lpai-dione/ssai/cvg/team/didonglin/lhz/AIXM-Video/aixmvideo/result/img/1.png')
            # save_image(zframe2,'/mnt/lpai-dione/ssai/cvg/team/didonglin/lhz/AIXM-Video/aixmvideo/result/img/2.png')
            x = vae.decode(z/0.18215).sample
            print(x.shape)
            video = rearrange(x, "(b f) c h w -> b f h w c", f=args.num_frames)
            video = video.squeeze(0)
            print(video.shape)
            video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
            imageio.mimwrite(args.save_video_path + str(i) + '.mp4', video, fps=8, quality=9)
            i = i+1

        elif args.dataset == "video_only":
            video = data.to(device)
            x = rearrange(video,'b c f h w -> (b f) c h w').contiguous()
            # sample 从目标分布中采样得到一个样本,vae.encode(x).latent_dist编码得到目标分布（均值、方差）,sample()根据分布采样隐空间向量
            z = vae.encode(x).latent_dist.sample().mul(0.18215)
            # 这里的sample只是将结果变为Tensorfloat, 不涉及采样
            x = vae.decode(z/0.18215).sample
            video = rearrange(x, "(b f) c h w -> b f h w c", f=args.num_frames)
            video = video.squeeze(0)
            print(video.shape)
            video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().contiguous()
            imageio.mimwrite(args.save_video_path + str(i) + '.mp4', video, fps=8, quality=9)
            i = i+1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_encoder_name", type=str, default="./hub/AI-ModelScope/t5-v1_1-xxl")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str,required=True)
    parser.add_argument("--video_folder", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_image_size", type=int, default=128)
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
        )
    parser.add_argument(
        "--model_max_length", type=int, default=120)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--sample_rate", type=int, default=4) 
    parser.add_argument("--save_video_path", type=str,required=True)
    parser.add_argument("--ae", type=str,required=True)


    args = parser.parse_args()
    main(args)