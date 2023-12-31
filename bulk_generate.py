import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
import argparse
import qrcode

# create input args for url and text prompt
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://youtu.be/dQw4w9WgXcQ", help="url of the image")
    parser.add_argument("--prompt", type=str, default="futuristic city, neon lights, matrix, stars in skyline, qr code, tetrics, unreal engine, artstation, detailed landscape, hd, james turrell lighting, a black and white photo of a qr code, cgtrader", help="text prompt")
    parser.add_argument("--negative_prompt", type=str, default="ugly, disfigured, low quality, blurry, nsfw", help="negative text prompt")
    # model from huggingface
    parser.add_argument("--model", type=str, default="stablediffusionapi/deliberate-v2", help="model from huggingface")
    # min and max values for conditioning_scale, guidance_scale and strength
    parser.add_argument("--min_cscale", type=float, default=0.7, help="min conditioning_scale")
    parser.add_argument("--max_cscale", type=float, default=2, help="max conditioning_scale")
    parser.add_argument("--min_gscale", type=float, default=15, help="min guidance_scale")
    parser.add_argument("--max_gscale", type=float, default=20, help="max guidance_scale")
    parser.add_argument("--min_strength", type=float, default=0.65, help="min strength")
    parser.add_argument("--max_strength", type=float, default=0.69, help="max strength")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # get input args
    args = parse_args()

    # 
    controlnet = ControlNetModel.from_pretrained("DionTimmer/controlnet_qrcode-control_v1p_sd15",
                                                torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model,        
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )

    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    def resize_for_condition_image(input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img

    print("Generating QR Code from content")
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(args.url)
    qr.make(fit=True)
    qrcode_image = qr.make_image(fill_color="black", back_color="white")
    qrcode_image.save("qr_code.png")

    # qr code image
    source_image = load_image("qr_code.png")
    # initial image, could be anything or qr code 
    init_image = load_image("qr_code.png")

    condition_image = resize_for_condition_image(source_image, 768)
    init_image = resize_for_condition_image(init_image, 768)
    generator = torch.manual_seed(123121231)

    # loop over conditioning_scale, guidance_scale and strength to get a valid QR Code Image
    for cscale in np.linspace(args.min_cscale,args.max_cscale,10):
        for gscale in np.linspace(args.min_gscale,args.max_gscale,10):
            for strength in np.linspace(args.min_strength,args.max_strength,10):
                image = pipe(prompt=args.prompt,
                            negative_prompt=args.negative_prompt,
                            image=init_image,
                            control_image=condition_image,
                            width=768,
                            height=768,
                            guidance_scale=gscale,
                            controlnet_conditioning_scale=cscale,
                            generator=generator,
                            strength=strength,
                            num_inference_steps=50,
                            )

                # save image to disk
                image[0][0].save(f"qr_code_{cscale:.2f}_{gscale:.2f}_{strength:.2f}.png")
