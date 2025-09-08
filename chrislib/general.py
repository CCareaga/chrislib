import math
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize

from chrislib.data_util import np_to_pil

def rescale(img, scale, r32=False):
    """resize an image according to a given scale

    params:
        img (numpy.array): an image as a np.array (can be HxW or HxWxC)
        scale (float): scale with which to resize the image dimensions

    returns:
        (numpy.array): the scaled image
    """
    h, w = img.shape[:2]
    round_func = round_32 if r32 else int
    return resize(img, (round_func(h * scale), round_func(w * scale)), preserve_range=True)

def pad_bb(bounding_box, amount=7):
    """Add padding to all elements of a PIL ImageDraw text bounding box.

    params:
        bounding_box (tuple): the bounding box to add padding to - must have four integer elements:
            left, top, right, bottom
        amount (int) optional: the amount of padding to add (default 7)

    returns:
        (tuple): bounding box tuple with padding added
    """
    return (
        bounding_box[0] - amount,
        bounding_box[1] - amount,
        bounding_box[2] + amount,
        bounding_box[3] + amount
    )


def add_chan(img):
    """Add an channel dimension to convert a gray-scale image 
    to an RGB image

    params:
        img (numpy.array): numpy array image (H x W)

    returns:
        (numpy.array): the numpy image with an added channel dimension (H x W x C)
    """

    # if the provided image already has a channel dim, assume it's a grayscale
    # image and make it HxW by grabbing the first channel
    if len(img.shape) == 3:
        img = img[:, :, 0]

    return np.stack([img] * 3, -1)


def _tile_row(images, text, border, font, font_pos, font_color, text_box):
    """helper function to for tile_imgs and show that creates a single row 
    of tiled images 

    params:
        images (list of np.array): array of images to tile
        text (list of string): list of strings to draw on each image
        border (int): the size of border to add between each image
        font (TODO): font to use to render text on images
        font_pos (tuple): the (x,y) coordinates to anchor the text relative to top-left (2 integers)
        font_color (tuple): the RGB values for the desired color (3 integers)
        text_box (int): opacity of text box around text, None for no text box

    returns:
        concat (np.array): row of images as a single tiled image
    """
    assert(len(images) == len(text))

    border_pix = np.ones((images[0].shape[0], border, 3))
    elements = [border_pix]

    for img, txt in zip(images, text):
        if len(img.shape) == 2:
            img = add_chan(img)

        if img.shape[-1] == 1:
            img = np.concatenate([img] * 3, -1)

        if txt !=  "" and txt is not None:
            pil_img = np_to_pil(img)
            draw = ImageDraw.Draw(pil_img, "RGBA")

            txtbb = draw.textbbox(font_pos, txt, font=font)

            if text_box is not None:
                draw.rectangle(pad_bb(txtbb), fill=(0, 0, 0, text_box))

            if font_color is None:
                imgbb = img[txtbb[1]:txtbb[3], txtbb[0]:txtbb[2], :]
                brightness = imgbb.mean(-1).mean()
                font_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)

            draw.text(font_pos, txt, font_color, font=font)
            img = np.array(pil_img) / 255.

        elements.append(img)
        elements.append(border_pix)

    concat = np.concatenate(elements, axis=1)
    return concat


def tile_imgs(
        images,
        rescale=1.0,
        text=None,
        font_size=16,
        font_file="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        font_color=None,
        font_pos=(0, 0),
        text_box=None,
        display=False,
        save=None,
        border=20,
        cmap='viridis',
        quality=75):
    """Tile a set of numpy images into a grid with text and spaces in between.

    params:
        images (list of np.array): images to tile, can be 1D or 2D array (must be rectangular grid)
        rescale (float) optional: the desired amount to rescale the tiled images (default 1.0)
        text (array-like) optional: text to put on each image, must be same shape as image array (default None)
        font_size (int) optional: the desired size of the font (default 16)
        font_file (str) optional: a filename or path to a file containing a TrueType font
        font_color (tuple) optional: the RGB values for the desired color (3 integers) (default
            None)
        font_pos (tuple) optional: the (x,y) coordinates to anchor the text (2 integers) (default
            (0, 0))
        text_box (int) optional: opacity of text box around text, no text box if set to None (default None)
        display (bool) optional: whether to display the tiled images (uses matplotlib to show tiled images)
        save (str) optional: the filename or path to save the tiled images to. Use None to not save
            (default None)
        border (int) optional: the size of border to add to each image (default 20)
        cmap (str) optional: the colormap for matplotlib to use to map scalar data to colors
            (default "viridis")
        quality (int) optional: the image quality to save with. Minimum (worst) is 0, maximum
            (best) is 90 (default 75)

    returns:
        tiled (np.array): single image of the tiled image set
    """
    if not isinstance(images, list):
        print("expected list of images")
        return None

    # make 1d array in 2d to keep logic simple
    if not isinstance(images[0], list):
        images = [images]

    # if text is none make a 2d array like images to make things
    # easier, otherwise text should already be shaped like images
    if text is None:
        text = []
        for row in images:
            text.append([None] * len(row))
    else:
        if not isinstance(text[0], list):
            text = [text]

    try:
        font = ImageFont.truetype(font_file, font_size)
    except:
        font = None

    width = sum([border + x.shape[1] for x in images[0]]) + border
    border_pix = np.ones((border, width, 3))
    rows = [border_pix]

    for img_row, txt_row in zip(images, text):
        tiled_row = _tile_row(img_row, txt_row, border, font, font_pos, font_color, text_box)

        rows.append(tiled_row)
        rows.append(border_pix)

    tiled = np.concatenate(rows, axis=0)

    if rescale != 1.0:
        h, w, _ = tiled.shape
        tiled = resize(tiled, (int(h * rescale), int(w * rescale)))

    if display:
        plt.figure(figsize=(len(images[0]), len(images)))
        plt.imshow(tiled, cmap=cmap)
        plt.axis('off')

    if save:
        # TODO: save tiled image
        byte_img = (tiled * 255.).astype(np.uint8)
        Image.fromarray(byte_img).save(save, quality=quality)
        # pass

    return tiled


def show(img, size=(16, 9), save=None):
    """Show a set of images as a tiled image using Matplotlib imshow (used in Jupyter notebooks)

    params:
        img (list of np.array): set of images to tile (can be 1D or 2D, must be rectangular)
        size (tuple) optional: the integer width and height values (default (16,9))
        save (str) optional: the filename or path to save the tiled images to. Use None to not save
            (default None)
    """
    if isinstance(img, list):
        img = tile_imgs(img, save=save)

    plt.figure(figsize=size)
    plt.imshow(img)
    plt.axis('off')


def match_scale(pred, grnd, mask=None, skip_close=False, threshold=0.001, subsample=1.0):
    """Match the scale between two arrays using least-squares

    params:
        pred (np.array): numpy array to be scaled
        grnd (np.array): numpy array to determine the scale
        mask (np.array) optional: values to include in the least-squares computation (default None)
        skip_close (bool) optional: whether or not to avoid scaling if determined scale is already close to 1 (default False)
        threshold (float) optional: threshold to determine if scale is close to 1 (default 0.001)
        subsample (float) optional: porportion of pixels to subsample for least-squares computation (default 1.0)

    returns:
        (np.array): the input pred array scaled to most closely match grnd
    """
    if mask is None:
        mask = np.ones(pred.shape[:2]).astype(bool)

    if subsample != 1.0:
        rand_mask = np.random.randn(*mask.shape) > (1.0 - subsample)
        mask &= rand_mask

    flat_pred = pred[mask].reshape(-1)
    flat_grnd = grnd[mask].reshape(-1)

    scale, _, _, _ = np.linalg.lstsq(flat_pred.reshape(-1, 1), flat_grnd, rcond=None)

    if skip_close and abs(1.0 - scale) < threshold:
        return pred

    return scale * pred


def get_scale(a, b, mask=None, subsample=1.0):
    """Function to find scale that most closely matches a to b

    params:
        a (np.array): array to be scaled
        b (np.array): array to determine the scale
        mask (np.array) optional: mask denoting pixels that should be used in the computation (default None)
        subsample (float) optional: porportion of pixels to subsample for least-squares computation (default 1.0)

    returns:
        scale (float): scale value x that minimizes sum(|a*x - b|^2)
    """
    if mask is None:
        mask = np.ones(pred.shape[:2])

    mask = mask.astype(bool)

    if subsample != 1.0:
        rand_mask = np.random.randn(*mask.shape) > (1.0 - subsample)
        mask &= rand_mask

    mask = mask.reshape(-1)
    flat_a = a.reshape(-1)[mask]
    flat_b = b.reshape(-1)[mask]

    scale, _, _, _ = np.linalg.lstsq(a.reshape(-1, 1), b.reshape(-1, 1), rcond=None)

    return scale


def get_brightness(rgb, mode='numpy', keep_dim=True):
    """use the CCIR601 YIQ method to compute brightness of an RGB image

    params:
        rgb (np.array or torch.Tensor): RGB image to convert to luminance
        mode (str) optional: whether the input is numpy or torch (default "numpy")
        keep_dim (bool) optional: whether or not to maintain the channel dimension

    returns:
        brightness (np.array or torch.Tensor): single channel image of brightness values
    """
    # "CCIR601 YIQ" method for computing brightness
    if mode == 'numpy':
        brightness = (0.3 * rgb[:,:,0]) + (0.59 * rgb[:,:,1]) + (0.11 * rgb[:,:,2])
        if keep_dim:
            brightness = brightness[:, :, np.newaxis]

    if mode == 'torch':
        brightness = (0.3 * rgb[0,:,:]) + (0.59 * rgb[1,:,:]) + (0.11 * rgb[2, :,:])
        if keep_dim:
            brightness = brightness.unsqueeze(0)

    return brightness

def minmax(img):
    """Min-Max normalize an image to put it in the [0-1] domain

    params:
        img (np.array or torch.Tensor): image to be normalized

    returns:
        img (np.array or torch.Tensor): min-max normalized image
    """
    return (img - img.min()) / img.max()


# def inv_2_real(inv_shd):
#     """TODO DESCRIPTION
# 
#     params:
#         inv_shd (TODO): TODO
# 
#     returns:
#         shd (TODO): TODO
#     """
#     # first normalize the network inverse shading to be [0,1]
#     norm_inv_shd = minmax(inv_shd)
# 
#     # convert to regular shading, clip very small values for division
#     shd = (1.0 / norm_inv_shd.clip(1e-5)) - 1.0
# 
#     return shd.clip(1e-5)


def round_32(x):
    """Round a number up to the next multiple of 32.

    params:
        x (numeric): a number to round

    returns:
        (int): x rounded up to the next multiple of 32
    """
    return 32 * math.ceil(x / 32)


def round_128(x):
    """Round a number up to the next multiple of 128.

    params:
        x (numeric): the number to round

    returns:
        (int): x rounded up to the next multiple of 128
    """
    return 128 * math.ceil(x / 128)


def to2np(img):
    """Convert a torch image with dimensions (channel, height, width) to a
    numpy image with dimensions (height, width, channel).

    params:
        img (torch.Tensor): a torch image with dimensions (c, h, w)

    returns:
        img (numpy.array): the image converted to numpy (h, w, c)
    """
    return img.detach().cpu().permute(1, 2, 0).numpy()


def view_scale(img, p=100):
    """Scale an image for visulization by normalizing based on a percentile.
    useful for visualizing shading images as it removes outliers.

    params:
        img (np.array): image to be normalized
        p (int) optional: percentile to compute max value (default 100)

    returns:
        img (numpy.array): image after diving by p-th percentile largest value and clipping to [0-1]
    """
    return (img / np.percentile(img, p)).clip(0, 1)


def view(img, p=100):
    """Performs p-th percentile scaling after gamma-correcting the image with a
    gamma value of 2.2. Basically the view_scale function for linear RGB values

    params:
        img (np.array): image to be scaled for viewing
        p (int) optional: percentile to compute the largest value (default 100)

    returns:
        img (np.array): image after gamma-correcting, scaling and clipping to [0-1]
    """
    return view_scale(img ** (1/2.2), p=p)


def invert(x):
    """Convert regular shading values to the "inverse shading" domain. More specifically
    performs the computation 1.0 / (x + 1.0) on a numpy array.

    params:
        x (np.array): values to convert to inverse space

    returns:
        out (np.array): input array with inverted values
    """
    out = 1.0 / (x + 1.0)
    return out


def uninvert(x, eps=0.001, clip=True):
    """Convert inverse shading values to back to the regular shading space.
    Reverse of the "invert" function. Computed (1.0 / x) - 1.0. Clips values to a
    minimum of epsilon to avoid divide-by-zero

    params:
        x (np.array): values to be uninverted
        eps (float) optional: minimum value allowed in x (default 0.001)
        clip (bool) optional: whether or not to clip small values (and zeros) in x (default True)

    returns:
        out (np.array): uninverted version of input x
    """
    if clip:
        x = x.clip(eps, 1.0)

    out = (1.0 / x) - 1.0
    return out


def get_tonemap_scale(rgb_color, p=90, target_brightness=0.8):
    """Compute the tonemapping scale for an HDR image following the CGIntrinsics 
    and Hypersim code-bases. The scale is determined such that the p-th percentile
    value in the input is equal to 0.8 after performing tonemapping.

    params:
        rgb_color (np.array): input rgb values to compute tonemap scale
        p (int) optional: percentile value to map to target brightness (default 90)
        target_brightness (float) optional: brightness value to map p-th percentile to (default 0.8)

    returns:
        scale (float): scale to multiple by the input before gamma-correction 
    """
    gamma = 1.0 / 2.2 # standard gamma correction exponent
    inv_gamma = 1.0 / gamma

    brightness       = get_brightness(rgb_color)
    # brightness_valid = brightness[valid_mask]

    # if the kth percentile brightness value in the unmodified image is less than this,
    # set the scale to 0.0 to avoid divide-by-zero
    eps = 0.0001
    brightness_nth_percentile_current = np.percentile(brightness, p)

    if brightness_nth_percentile_current < eps:
        scale = 0.0
    else:
        # Snavely uses the following expression in the code at
        # https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
        # scale = np.exp(
        #           np.log(
        #               target_brightness) *
        #               inv_gamma -
        #               np.log(brightness_nth_percentile_current))
        #
        # Our expression below is equivalent, but is more intuitive, because it follows more
        # directly from the expression:
        # (scale*brightness_nth_percentile_current)^gamma = target_brightness
        # pylint: disable-next=line-too-long
        scale = np.power(target_brightness, inv_gamma) / brightness_nth_percentile_current

    return scale
