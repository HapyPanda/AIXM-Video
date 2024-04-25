import urllib.parse as ul

def get_precision(args):
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dtype

def text_preprocessing(text):
    # The exact text cleaning as was in the training stage:
    text = clean_caption(text)
    text = clean_caption(text)
    return text

