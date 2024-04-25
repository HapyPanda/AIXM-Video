from .transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo
from transformers import AutoTokenizer
from .t2v_datasets import T2V_dataset
from .video_only_datasets import Videodataset
from torchvision.transforms import Lambda
from torchvision import transforms

ae_norm = {
    'CausalVAEModel_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVQVAEModel_4x4x4': Lambda(lambda x: x - 0.5),
    'CausalVQVAEModel_4x8x8': Lambda(lambda x: x - 0.5),
    'VQVAEModel_4x4x4': Lambda(lambda x: x - 0.5),
    'VQVAEModel_4x8x8': Lambda(lambda x: x - 0.5),
    "bair_stride4x2x2": Lambda(lambda x: x - 0.5),
    "ucf101_stride4x4x4": Lambda(lambda x: x - 0.5),
    "kinetics_stride4x4x4": Lambda(lambda x: x - 0.5),
    "kinetics_stride2x4x4": Lambda(lambda x: x - 0.5),
    'stabilityai/sd-vae-ft-mse': transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    'stabilityai/sd-vae-ft-ema': transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    'vqgan_imagenet_f16_1024': Lambda(lambda x: 2. * x - 1.),
    'vqgan_imagenet_f16_16384': Lambda(lambda x: 2. * x - 1.),
    'vqgan_gumbel_f8': Lambda(lambda x: 2. * x - 1.),
}

ae_denorm = {
    'CausalVAEModel_4x8x8': lambda x: (x + 1.) / 2.,
    'CausalVQVAEModel_4x4x4': lambda x: x + 0.5,
    'CausalVQVAEModel_4x8x8': lambda x: x + 0.5,
    'VQVAEModel_4x4x4': lambda x: x + 0.5,
    'VQVAEModel_4x8x8': lambda x: x + 0.5,
    "bair_stride4x2x2": lambda x: x + 0.5,
    "ucf101_stride4x4x4": lambda x: x + 0.5,
    "kinetics_stride4x4x4": lambda x: x + 0.5,
    "kinetics_stride2x4x4": lambda x: x + 0.5,
    'stabilityai/sd-vae-ft-mse': lambda x: 0.5 * x + 0.5,
    'stabilityai/sd-vae-ft-ema': lambda x: 0.5 * x + 0.5,
    'vqgan_imagenet_f16_1024': lambda x: (x + 1.) / 2.,
    'vqgan_imagenet_f16_16384': lambda x: (x + 1.) / 2.,
    'vqgan_gumbel_f8': lambda x: (x + 1.) / 2.,
}

def getdataset(args):
    # 随机返回连续的目标帧数视频的（开始idx，结尾idx）帧
    temporal_sample = TemporalRandomCrop(args.num_frames * args.sample_rate)  # 16 x
    norm_fun = ae_norm[args.ae]

    if args.dataset == 'video_only':
        transform = transforms.Compose(
            [
                ToTensorVideo(),  # TCHW
                # CenterCropResizeVideo(size=args.max_image_size),
                # RandomHorizontalFlipVideo(p=0.5),
                norm_fun,
            ]
        )
        return Videodataset(args, transform=transform, temporal_sample=temporal_sample)
    # elif args.dataset == 'sky':
    #     transform = transforms.Compose([
    #         ToTensorVideo(),
    #         CenterCropResizeVideo(args.max_image_size),
    #         RandomHorizontalFlipVideo(p=0.5),
    #         norm_fun
    #     ])
    #     return Sky(args, transform=transform, temporal_sample=temporal_sample)

    if args.dataset == 't2v':
        transform = transforms.Compose([
            # 将视频数据转为tensor,video/255
            ToTensorVideo(),
            # 中心裁剪＋resize
            CenterCropResizeVideo(args.max_image_size),
            # 随机翻转视频帧
            # RandomHorizontalFlipVideo(p=0.5),
            norm_fun
        ])
        tokenizer = AutoTokenizer.from_pretrained("/mnt/lpai-dione/ssai/cvg/team/didonglin/lhz/hub/AI-ModelScope/t5-v1_1-xxl")
        return T2V_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer)