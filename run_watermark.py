"""
Script to run the watermark.
"""

import PIL
import PIL.Image
import torch
import torch.nn.functional as F

import pandas as pd

from utils.wm.wm_utils import WmProviders
from utils.wm.gs_provider import parser as gs_parser
from utils.wm.tr_provider import parser as tr_parser
from utils.wm.prc_provider import parser as prc_parser
from utils.wm.tag_provider import parser as tag_parser
from utils.wm.ringid_provider import parser as ringid_parser
from utils.wm.hstr_provider import parser as hstr_parser
from utils.wm.hsqr_provider import parser as hsqr_parser

from utils.pipe import pipe_utils
from utils.imprint_utils import invert_image, validate
from utils.image_utils import distort_images, check_flag
from utils.finger_utils import decode_tensors, decode_tensors_reverse#, spd_dist_latents, matched_spd_distance
from utils.utils import get_detection_threshold, check_if_detection_successful
from utils.utils import set_random_seed, seed_everything

from utils.prompt_utils import get_text_prompts

import os
from tqdm import tqdm
from utils.logger import get_logger


model_id = ["CompVis/stable-diffusion-v1-4",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1-base",
            "stabilityai/stable-diffusion-xl-base-1.0", 
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", 
            "cagliostrolab/animagine-xl-3.0",
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            "THUDM/CogView4-6B"]

model_flux = ["black-forest-labs/FLUX.1-dev",
              "stabilityai/stable-diffusion-3-medium-diffusers",
              "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
              "THUDM/CogView4-6B"]

model_name_mapping = {
    "CompVis/stable-diffusion-v1-4": "sd14",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "sd15",
    "stabilityai/stable-diffusion-3-medium-diffusers": "sd3",
    "THUDM/CogView4-6B": "cogview4",
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS": "pixart",
    "PixArt-alpha/PixArt-XL-2-512x512": "pixart-xl",
    "black-forest-labs/FLUX.1-dev": "flux",
    "stabilityai/stable-diffusion-2-1-base": "sd21",
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers": "sana",
}

parent_parsers = [
    gs_parser, tr_parser, prc_parser, 
    tag_parser, ringid_parser, hstr_parser,
    hsqr_parser
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args
import argparse
parser = argparse.ArgumentParser(description="test_watermark", parents=parent_parsers)

parser.add_argument("--out_dir", type=str, default="out/watermark_gen/")
parser.add_argument("--target_prompt", type=str, default="cat standing on a rock in front of a crowd of cats, backlighting, digital art, trending on pixiv, fanart")

# target model
parser.add_argument("--modelid_target",
                    type=str,
                    default="stabilityai/stable-diffusion-xl-base-1.0",
                    choices=[model for model in model_id])
parser.add_argument("--scheduler_target", type=str, default="DDIM")
parser.add_argument("--num_inference_steps_target", type=int, default=50)  # 20 for FLUX, 28 for SD3, 20 for Pix
parser.add_argument("--guidance_scale_target", type=float, default=7.5)  # 3.5 for FLUX, 7 for SD3, 4.5 for Pix
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--wm_type",
                    type=str,
                    default="GS",
                    choices=[wm.name for wm in WmProviders])
parser.add_argument("--distort", action="store_true", default=False)

# dataset
parser.add_argument("--dataset_id", type=str, default="Gustavo", choices=["Gustavo", "coco", "DB1k"])

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num", type=int, default=100)
parser.add_argument("--logger", action="store_true", default=False)
parser.add_argument("--save", action="store_true", default=False)

args, unknown_args = parser.parse_known_args()

# set seeds
set_random_seed(args.seed)

# logger
if args.logger:
    log_name = f"{model_name_mapping[args.modelid_target]}_{args.wm_type}"
    logger = get_logger(args.out_dir, log_name)

# retrieve the detection threshold for the settings
detection_threshold = get_detection_threshold(args.wm_type, args.modelid_target)
target_prompts = get_text_prompts(num_prompts=args.num, dataset_id=args.dataset_id)
# target_prompts = target_prompts[1:]

# pipe_provider used by the target model (SDXL, PixArt, FLUX)
pipe_provider_target = pipe_utils.get_pipe_provider(pretrained_model_name_or_path=args.modelid_target,
                                                    resolution=args.resolution,
                                                    device=DEVICE,
                                                    eager_loading=True if args.modelid_target in model_flux else False,
                                                    disable_tqdm=True,)

# generate a watermarked latent zT
# This way like it is done here is a simple way to obtain a watermark provider for a simple test run.
# If you want to do mass experiments and have batch_sizes > 1, plz have look at the utils.wm_provider.WmProvider.generate_providers method
wm_provider = WmProviders[args.wm_type].value(latent_shape=pipe_provider_target.get_latent_shape(), **vars(args))
wm_initial_results = wm_provider.get_wm_latents()
wm_zT = wm_initial_results["zT_torch"]

# for Gaussian Shading, we also get an initial message
message_bits_str_initial = wm_initial_results["message_bits_str_list"][0] if "message_bits_str_list" in wm_initial_results else None

metric_map = {
    "PRC": "value",
    "TR": "p_value",
    "RID": "l1_dist",
    "HSTR": "l1_dist",
    "HSQR": "l1_dist",
    "GS": "bit_accuracy",
    "TAG": "bit_accuracy",
}

wm_list = []
ret_list = []

results_data = []
header = ["inter_max", "inter_mean", "inter_std", "inter_topk_mean", "intra_max", "intra_mean", "intra_std", ]
all_results = []
for id, (target_prompt) in tqdm(enumerate(target_prompts), total=len(target_prompts), desc="Generating images"):
    print(f"\n--- Starting run {id+1}/{len(target_prompts)} ---")
    # print(f"Target prompt: {target_prompt}")
    if args.logger:
        logger.info("Test single watermarked image")

    if args.wm_type in ["PRC"]:
        seed_everything(args.seed)

    # generate a watermarked image with the target model

    generated_PIL_list = pipe_provider_target.generate(prompts=target_prompt,
                                                    latents=wm_zT,
                                                    num_inference_steps=args.num_inference_steps_target,
                                                    guidance_scale=args.guidance_scale_target,
                                                    # callback_on_step_end=decode_tensors,
                                                    # callback_on_step_end_tensor_inputs=["latents"]
                                                    )["images_PIL"]
    benign_image = generated_PIL_list[0]

    # benign_image.save("watermarked_image_TR.png")

    # from PIL import Image
    # benign_image = Image.open("sana_vae_TR.png").convert("RGB")

    if args.distort:
        benign_image = distort_images(benign_image, rba_delta=1.0)
        benign_image.save("distort_image.png")

    # distort param:
    # r_degree=(0, 150), jpeg_ratio=(10, 90), sp_prob_fixed=(0.05, 0.4), crop_scale_TR=(0.9 raw, 0.5),
    # random_crop_ratio=(0.1-0.5), random_drop_ratio=(0.1, 0.8), gaussian_std_fixed=(0.05, 0.4), :done 
    # median_blur_k=(1,3,5,...,17), gaussian_blur_r=(2,4,6,8,10), resize_ratio=(0.1, 0.9),
    # brightness_factor=(2,16), contrast_factor, 
    # vertical_shift_ratio=(0.1, 0.8), horizontal_shift_ratio=(0.1, 0.8), flip_ratio=1,

    with torch.no_grad(): # retrieve zT
        zT_retrieved = pipe_provider_target.invert_images(benign_image, num_inference_steps=args.num_inference_steps_target)["zT_torch"]
    # mse = F.mse_loss(wm_zT, zT_retrieved)
    # print("mse:", mse.item())
    
    # from utils.spd_utils import spd_dist_latents, matched_spd_distance
    # d_corr = spd_dist_latents(wm_zT, zT_retrieved)
    # # print("SPD distance:", d_corr)
    # d_auto, sigma = matched_spd_distance(wm_zT, zT_retrieved)
    # d_final = min(d_corr, d_auto)
    # print("SPD distance (corr):", d_final)

    # if args.logger:
    #     logger.info(f"SPD distance (corr): {d_final}")
    # else:
    #     print("SPD distance (corr):", d_final)

    # from utils.spd_utils import multi_spd_airm_dist_latents
    # d2 = multi_spd_airm_dist_latents(wm_zT, zT_retrieved, n_parts=4)
    # print("multi-patch:", d2)
    # if args.logger:
    #     logger.info(f"SPD distance (corr): {d_corr}")
    # else:
    #     print("SPD distance (corr):", d_corr)
    # print("blur distance (corr):", d_auto, sigma)
    
    # if intra_score['patch_mean_std'] < 0.125:
    #     if d_final < 0.37:
    #         result = "No Detection"
    #     else:
    #         result = "Detect Forgery"
    # else:
    #     result = "Detect Forgery"

    # results_data.append(data)

    # from utils.spd_utils import spectral_curv
    # d = spectral_curv(zT_retrieved)
    # print("Slope", d['slope'])
    # print("Noise Ratio", d['noise_ratio'])

    # from utils.spd_utils import latent_cosine
    # cos_sim = latent_cosine(wm_zT, zT_retrieved)
    # print("Latent Cos Sim:", cos_sim)

    ######### save latent. #########
    # wm_zT_cpu = wm_zT.detach().cpu().to(torch.float32)
    # zT_ret_cpu = zT_retrieved.detach().cpu().to(torch.float32)

    # wm_list.append(wm_zT_cpu)
    # ret_list.append(zT_ret_cpu)

    # from utils.spd_utils import local_spd_features, local_airm_score
    # feat_w = local_spd_features(wm_zT)
    # feat_r = local_spd_features(zT_retrieved)
    # # score, ratio  = local_airm_score(feat_w, feat_r)
    # score = local_airm_score(feat_w, feat_r)
    # print("Local AIRM score:", score)

    # if score > 1.3:
    #     if ratio > 0.23:
    #         if d_final > 0.3:
    #             # logger.info("Detect Forgery")
    #             print("Detect Forgery")
    #         else:
    #             # logger.info("No Forgery Detected")
    #             print("No Forgery Detected")
    #     else:
    #         # logger.info("No Forgery Detected")
    #         print("No Forgery Detected")
    # else:
    #     # logger.info("No Forgery Detected")
    #     print("No Forgery Detected")

    # from utils.finger_utils import spectral_similarity
    # sim = spectral_similarity(wm_zT, zT_retrieved)
    # print("Spectral similarity:", sim)
    
    # benign_image = PIL.Image.open("watermarked_image.png")

    rows = []
    results = validate(
            out_dir=args.out_dir,
            image_to_verify_PIL=benign_image,
            original_PIL=benign_image,
            wm_provider=wm_provider,
            pipe_provider_target=pipe_provider_target,
            num_inference_steps_target=args.num_inference_steps_target,
            step=-1,
            message_bits_str_initial=message_bits_str_initial,
            do_psnr=False,
            do_ssim=False,
            do_msssim=False,
            do_lpips=False,
            # callback_on_step_end=decode_tensors_reverse,
            # callback_on_step_end_tensor_inputs=["latents"]
            )

    # check if detection was successfull
    detection_successful = check_if_detection_successful(wm_type=args.wm_type,
                                                        threshold=detection_threshold,
                                                        value=results[metric_map[args.wm_type]])
    results["detection_successful"] = detection_successful
    
    # base_path = "./latent_hash"
    # tmp_flag_path = os.path.join(base_path, "tmp_flag.txt")
    # flag_status = check_flag(tmp_flag_path)
    
    # pool_path = os.path.join(base_path, "hash_pool.npy")
    # os.remove(pool_path)
    
    # if not flag_status:
    #     import sys
    #     # results["detection_successful"] = False
    #     sys.exit(0)

    rows.append({
                "bit_accuracy": results["bit_accuracy"],
                "p_value": results["p_value"],
                "value": results["value"],
                "detection_success": results["detection_success"],
                "log_message": results["log_message"],
                "prc_threshold": results["prc_threshold"],
                "detection_successful": results["detection_successful"],
                })
    if args.logger:
        logger.info(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {results['bit_accuracy']:.5f}, p_value: {results['p_value']}, PRC value: {results['value']:.5f}, l1_dist: {-results['l1_dist']:.5f}")
    else:
        print(f"(Benign image) detection_success: {detection_successful}, bit accuracy: {results['bit_accuracy']:.5f}, p_value: {results['p_value']}, PRC value: {results['value']:.5f}, l1_dist: {-results['l1_dist']:.5f}")
    
    all_results.extend(rows)

# wm_latents  = torch.cat(wm_list,  dim=0)   # [N * B, C, H, W]
# ret_latents = torch.cat(ret_list, dim=0)   # [N * B, C, H, W]

# torch.save(
#     {
#         "wm_latents": wm_latents,
#         "ret_latents": ret_latents,
#     },
#     f"{args.wm_type}_distort_latents_gsblur12.pt",
# )

# import numpy as np
# Sigmas_wm = np.stack(Sigma_list, axis=0)  # [N, C, C]
# np.save("wm_spd.npy", Sigmas_wm)

# import csv
# filename = f"dual_{args.wm_type}_record_num_{len(target_prompts)}.csv"
# with open(os.path.join(args.out_dir, filename), 'w', newline='', encoding='utf-8') as file:
#     # writer = csv.writer(file)
#     # writer.writerow(header)
#     writer = csv.DictWriter(file, fieldnames=header)
#     writer.writeheader()
#     writer.writerows(results_data)
# print(f"\n save csv {filename}")

if args.save:
    df = pd.DataFrame(all_results)
    filename = f"{args.wm_type}_num_{len(target_prompts)}.csv"
    df.to_csv(os.path.join(args.out_dir, filename))