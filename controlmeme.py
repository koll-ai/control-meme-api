from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from share import *

model = None
ddim_sampler = None

def load_model(cn_model, location="cuda"):
    global model
    global ddim_sampler

    #free the GPU cache
    model = None
    torch.cuda.empty_cache()

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(cn_model, location=location))
    
    if location=="cuda":
      model = model.cuda()
    
    ddim_sampler = DDIMSampler(model)

#TODO check if working 
def transfer_control(other_model, cn_model):
    global model
    path_sd15 = './models/v1-5-pruned.ckpt'
    path_sd15_with_control = cn_model
    path_input = other_model

    sd15_state_dict = load_state_dict(path_sd15)
    sd15_with_control_state_dict = load_state_dict(path_sd15_with_control)
    input_state_dict = load_state_dict(path_input)


    def get_node_name(name, parent_name):
        if len(name) <= len(parent_name):
            return False, ''
        p = name[:len(parent_name)]
        if p != parent_name:
            return False, ''
        return True, name[len(parent_name):]


    keys = sd15_with_control_state_dict.keys()

    final_state_dict = {}
    for key in keys:
        is_first_stage, _ = get_node_name(key, 'first_stage_model')
        is_cond_stage, _ = get_node_name(key, 'cond_stage_model')
        if is_first_stage or is_cond_stage:
            final_state_dict[key] = input_state_dict[key]
            continue
        p = sd15_with_control_state_dict[key]
        is_control, node_name = get_node_name(key, 'control_')
        if is_control:
            sd15_key_name = 'model.diffusion_' + node_name
        else:
            sd15_key_name = key
        if sd15_key_name in input_state_dict:
            p_new = p + input_state_dict[sd15_key_name] - sd15_state_dict[sd15_key_name]
        else:
            p_new = p
        final_state_dict[key] = p_new

    model.load_state_dict(final_state_dict)

def generate(hint, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
      img = resize_image(HWC3(hint), image_resolution)
      
      H, W, C = img.shape

      control = torch.from_numpy(img.copy()).float().cuda() / 255.0
      control = torch.stack([control for _ in range(num_samples)], dim=0)
      control = einops.rearrange(control, 'b h w c -> b c h w').clone()

      if seed == -1:
          seed = random.randint(0, 65535)
      seed_everything(seed)

      if config.save_memory:
          model.low_vram_shift(is_diffusing=False)

      cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
      un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
      shape = (4, H // 8, W // 8)

      if config.save_memory:
          model.low_vram_shift(is_diffusing=True)

      model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
      samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

      if config.save_memory:
          model.low_vram_shift(is_diffusing=False)

      x_samples = model.decode_first_stage(samples)
      x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

      results = [x_samples[i] for i in range(num_samples)]

      return results
