import torch
import numpy as np
from skimage.transform import resize

from .ordinal_util import base_resize, equalize_predictions
from .resolution_util import optimal_resize
from .general import round_32


def run_pipeline(models, img_arr, output_ordinal=False, resize_conf=None, base_size=384, maintain_size=False, linear=False, device='cuda', lstsq_p=0.0, inputs='all'):
    """TODO DESCRIPTION

    params:
        * models (dict): models dictionary returned by load_models()
        * img_arr (np.array): RGB input image as numpy array between 0-1
        * output_ordinal (bool) optional: whether or not to output intermediate ordinal estimations (default False)
        * resize_conf (float) optional: confidence to use for resizing (between 0-1) if None maintain original size (default None)
        * base_size (int) optional: TODO (default 384)
        * maintain_size (bool) optional: whether or not the results match the input image size (default False)
        * linear (bool) optional: TODO (default False)
        * device (str) optional: string representing device to use for pipeline (default "cuda")
        * lstsq_p (float) optional: TODO (default 0.0)
        * inputs (str) optional: network inputs ("full", "base", "rgb", "all") the rgb image is always included (default "all")

    returns:
        * results (TODO): TODO
    """    
    results = {}
    
    orig_h, orig_w, _ = img_arr.shape
    
    if resize_conf == None:
        img_arr = resize(img_arr, (round_32(orig_h), round_32(orig_w)), anti_aliasing=True)

    elif isinstance(resize_conf, int):
        scale = resize_conf / max(orig_h, orig_w)
        img_arr = resize(img_arr, (round_32(orig_h * scale), round_32(orig_w * scale)), anti_aliasing=True)

    elif isinstance(resize_conf, float):
        img_arr = optimal_resize(img_arr, conf=resize_conf)

    fh, fw, _ = img_arr.shape
    # print(f"resized image to: ({fh}, {fw})")
    
    if not linear:
        lin_img = img_arr ** 2.2
    else:
        lin_img = img_arr
    
    with torch.no_grad():
        # ordinal shading estimation --------------------------
        base_input = base_resize(lin_img, base_size)
        full_input = lin_img

        base_input = torch.from_numpy(base_input).permute(2, 0, 1).to(device).float()
        full_input = torch.from_numpy(full_input).permute(2, 0, 1).to(device).float()

        base_out = models['ordinal_model'](base_input.unsqueeze(0)).squeeze(0)
        full_out = models['ordinal_model'](full_input.unsqueeze(0)).squeeze(0)
        
        base_out = base_out.cpu().numpy()
        full_out = full_out.cpu().numpy()

        base_out = resize(base_out, (fh, fw, 1))
        full_out = full_out[:, :, np.newaxis]
        
        # if we are using all inputs, we scale the input estimations using the base estimate
        if inputs == 'all':
            ord_base, ord_full = equalize_predictions(lin_img, base_out, full_out, p=lstsq_p)
        else:
            ord_base, ord_full = base_out, full_out
        # ------------------------------------------------------
       
        # ordinal shading to real shading ----------------------
        inp = torch.from_numpy(lin_img).permute(2, 0, 1).to(device)
        bse = torch.from_numpy(ord_base).permute(2, 0, 1).to(device)
        fll = torch.from_numpy(ord_full).permute(2, 0, 1).to(device)
        
        if inputs == 'full':
            combined = torch.cat((inp, fll), 0).unsqueeze(0)
        elif inputs == 'base':
            combined = torch.cat((inp, bse), 0).unsqueeze(0)
        elif inputs == 'rgb':
            combined = inp.unsqueeze(0)
        else:
            combined = torch.cat((inp, bse, fll), 0).unsqueeze(0)

        inv_shd = models['real_model'](combined)
        
        shd = ((1.0 / inv_shd) - 1.0)
        alb = inp / shd
        # ------------------------------------------------------
    
    inv_shd = inv_shd.squeeze(0).detach().cpu().numpy()
    alb = alb.permute(1, 2, 0).detach().cpu().numpy()

    if maintain_size:
        if output_ordinal:
            ord_base = resize(base_out, (orig_h, orig_w), anti_aliasing=True)
            ord_full = resize(full_out, (orig_h, orig_w), anti_aliasing=True)
            
        inv_shd = resize(inv_shd, (orig_h, orig_w), anti_aliasing=True)
        alb = resize(alb, (orig_h, orig_w), anti_aliasing=True)
    
    if output_ordinal:
        results['ord_full'] = ord_full
        results['ord_base'] = ord_base

    results['inv_shading'] = inv_shd
    results['albedo'] = alb
    results['image'] = img_arr

    return results
