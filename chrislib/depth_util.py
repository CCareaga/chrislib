import sys
from argparse import Namespace
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

CURR_PATH = '/home/chris/research/intrinsic/misc/boosted_depth'
# CURR_PATH = '/project/aksoy-lab/chris/misc/'
sys.path.append(CURR_PATH)

# OUR
from BoostingMonocularDepth.utils import calculateprocessingres

# MIDAS
from BoostingMonocularDepth.midas.models.midas_net import MidasNet
from BoostingMonocularDepth.midas.models.transforms import Resize, NormalizeImage, PrepareForNet

# PIX2PIX : MERGE NET
from BoostingMonocularDepth.pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

WHOLE_SIZE_THRESHOLD = 3000  # R_max from the paper
GPU_THRESHOLD = 1600 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted


def create_depth_models(device='cuda', midas_path=None, pix2pix_path=None):
    """TODO DESCRIPTION

    params:
        device (str) optional: TODO (default "cuda")
        midas_path (TODO) optional: TODO (default None)
        pix2pix_path (TODO) optional: TODO (default None)

    returns:
        (list): TODO
    """
    # opt = TestOptions().parse()
    opt = Namespace(
        Final=False,
        R0=False,
        R20=False,
        aspect_ratio=1.0,
        batch_size=1,
        checkpoints_dir=f'{CURR_PATH}/BoostingMonocularDepth/pix2pix/checkpoints',
        colorize_results=False,
        crop_size=672,
        data_dir=None,
        dataroot=None,
        dataset_mode='depthmerge',
        depthNet=None,
        direction='AtoB',
        display_winsize=256,
        epoch='latest',
        eval=True,
        generatevideo=None,
        gpu_ids=[0],
        init_gain=0.02,
        init_type='normal',
        input_nc=2,
        isTrain=False,
        load_iter=0,
        load_size=672,
        max_dataset_size=10000,
        max_res=float('inf'),
        model='pix2pix4depth',
        n_layers_D=3,
        name='mergemodel',
        ndf=64,
        netD='basic',
        netG='unet_1024',
        net_receptive_field_size=None,
        ngf=64,
        no_dropout=False,
        no_flip=False,
        norm='none',
        num_test=50,
        num_threads=4,
        output_dir=None,
        output_nc=1,
        output_resolution=None,
        phase='test',
        pix2pixsize=None,
        preprocess='resize_and_crop',
        savecrops=None,
        savewholeest=None,
        serial_batches=False,
        suffix='',
        verbose=False)
    # opt = Namespace()
    # opt.gpu_ids = [0]
    # opt.isTrain = False
    # global pix2pixmodel

    pix2pixmodel = Pix2Pix4DepthModel(opt)

    if pix2pix_path is None:
        # pylint: disable-next=line-too-long
        pix2pixmodel.save_dir = f'{CURR_PATH}/BoostingMonocularDepth/pix2pix/checkpoints/mergemodel'
    else:
        pix2pixmodel.save_dir = pix2pix_path

    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()

    if midas_path is None:
        midas_model_path = f"{CURR_PATH}/BoostingMonocularDepth/midas/model.pt"
    else:
        midas_model_path = midas_path

    # global midasmodel
    midasmodel = MidasNet(midas_model_path, non_negative=True)
    midasmodel.to(device)
    midasmodel.eval()

    return [pix2pixmodel, midasmodel]


def get_depth(img, models, threshold=0.2):
    """TODO DESCRIPTION

    params:
        img (TODO): TODO
        models (TODO): TODO
        threshold (float) optional: TODO (default 0.2)

    returns:
        whole_estimate (TODO): TODO
    """
    pix2pixmodel, midasmodel = models

    # Generate mask used to smoothly blend the local pathc estimations to the base estimate.
    # It is arbitrarily large to avoid artifacts during rescaling for each crop.
    # mask_org = generatemask((3000, 3000))
    # mask = mask_org.copy()

    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = threshold

    # print("start processing")

    input_resolution = img.shape

    scale_threshold = 3  # Allows up-scaling with a scale up to 3

    # Find the best input resolution R-x. The resolution search described in section 5-double
    # estimation of the main paper and section B of the supplementary material.
    whole_image_optimal_size, patch_scale = calculateprocessingres(
        img,
        384,
        r_threshold_value, scale_threshold,
        WHOLE_SIZE_THRESHOLD)

    # print('\t wholeImage being processed in :', whole_image_optimal_size)

    # Generate the base estimate using the double estimation.
    whole_estimate = doubleestimate(
        img,
        384,
        whole_image_optimal_size,
        1024,
        pix2pixmodel,
        midasmodel)
    whole_estimate = cv2.resize(
        whole_estimate,
        (input_resolution[1], input_resolution[0]),
        interpolation=cv2.INTER_CUBIC)

    return whole_estimate


# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pixsize, pix2pixmodel, midasmodel):
    """TODO DESCRIPTION

    params:
        img (TODO): TODO
        size1 (TODO): TODO
        size2 (TODO): TODO
        pix2pixsize (TODO): TODO
        pix2pixmodel (TODO): TODO
        midasmodel (TODO): TODO

    returns:
        prediction_mapped (TODO): TODO
    """
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, midasmodel)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, midasmodel)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


# Generate a single-input depth estimation
def singleestimate(img, msize, midasmodel):
    """TODO DESCRIPTION

    params:
        img (TODO): TODO
        msize (TODO): TODO
        midasmodel (TODO): TODO

    returns:
        (TODO): TODO
    """
    #if msize > GPU_THRESHOLD:
    #    print(" \t \t DEBUG| GPU THRESHOLD REACHED", msize, '--->', GPU_THRESHOLD)
    #    msize = GPU_THRESHOLD
    msize = min(msize, GPU_THRESHOLD)

    return estimatemidas(img, midasmodel, msize)
    # elif net_type == 1:
    #     return estimatesrl(img, msize)
    # elif net_type == 2:
    #     return estimateleres(img, msize)


def estimatemidas(img, midasmodel, msize, device='cuda'):
    """TODO DESCRIPTION

    params:
        img (TODO): TODO
        midasmodel (TODO): TODO
        msize (TODO): TODO
        device (str) optional: TODO (default "cuda")

    returns:
        prediction (TODO): TODO
    """
    # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2
    transform = Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodel.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(
        prediction,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction


def write_depth(path, depth, bits=1 , colored=False):
    """Write depth map to pfm and png file.

    params:
        path (str): filepath without extension
        depth (array): depth
        bits (int) optional: TODO (default 1)
        colored (bool) optional: TODO (default False)
    """
    # write_pfm(path + ".pfm", depth.astype(np.single))
    if colored is True:
        bits = 1

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1
    # if depth_max>max_val:
    #     print('Warning: Depth being clipped')
    #
    # if depth_max - depth_min > np.finfo("float").eps:
    #     out = depth
    #     out [depth > max_val] = max_val
    # else:
    #     out = 0

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1 or colored:
        out = out.astype("uint8")
        if colored:
            out = cv2.applyColorMap(out,cv2.COLORMAP_INFERNO)
        cv2.imwrite(path+'.png', out)
    elif bits == 2:
        cv2.imwrite(path+'.png', out.astype("uint16"))
