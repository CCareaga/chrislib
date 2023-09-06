import numpy as np
import math
import scipy as sp

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize

from chrislib.data_util import np_to_pil


def pad_bb(bounding_box, amount=7):
    """Add padding to all elements of a PIL ImageDraw text bounding box.

    params:
        * bounding_box (tuple): the bounding box to add padding to - must have four integer elements: left, top, right, bottom
        * amount (int) optional: the amount of padding to add (default 7)

    returns:
        * (tuple): bounding box tuple with padding added
    """
    return (
        bounding_box[0] - amount,
        bounding_box[1] - amount,
        bounding_box[2] + amount,
        bounding_box[3] + amount
    )


def add_chan(img):
    """Add a channel dimension to an image.

    params:
        * img (numpy.array):

    returns:
        * (numpy.array): the numpy image with an added channel dimension
    """
    return np.stack([img] * 3, -1)


def _tile_row(images, text, border, font, font_pos, font_color, text_box):
    """DESCRIPTION

    params:
        * images ():
        * text ():
        * border (int): the size of border to add to each image
        * font ():
        * font_pos (tuple): the (x,y) coordinates to anchor the text (2 integers)
        * font_color (tuple): the RGB values for the desired color (3 integers)
        * text_box ():

    returns:
        * concat ():
    """
    assert(len(images) == len(text))

    border_pix = np.ones((images[0].shape[0], border, 3))
    elements = [border_pix]

    for im, txt in zip(images, text):
        if len(im.shape) == 2:
            im = add_chan(im)

        if im.shape[-1] == 1:
            im = np.concatenate([im] * 3, -1)
       
        if txt !=  "" and txt is not None:
            pil_im = np_to_pil(im)
            draw = ImageDraw.Draw(pil_im, "RGBA")

            txtbb = draw.textbbox(font_pos, txt, font=font)
        
            if text_box is not None:
                draw.rectangle(pad_bb(txtbb), fill=(0, 0, 0, text_box))

            if font_color is None:
                imgbb = im[txtbb[1]:txtbb[3], txtbb[0]:txtbb[2], :]
                brightness = imgbb.mean(-1).mean()
                font_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)

            draw.text(font_pos, txt, font_color, font=font)
            im = np.array(pil_im) / 255.

        elements.append(im)
        elements.append(border_pix)

    concat = np.concatenate(elements, axis=1)
    return concat


def tile_imgs(images, rescale=1.0, text=None, font_size=16, font_file="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              font_color=None, font_pos=(0, 0), text_box=None, display=False, save=None, border=20, cmap='viridis', quality=75):
    """DESCRIPTION

    params:
        * images ():
        * rescale (float) optional: the desired amount to rescale the tiled images (default 1.0)
        * text (array-like) optional:
        * font_size (int) optional: the desired size of the font (default 16)
        * font_file (str) optional: a filename or path to a file containing a TrueType font
        * font_color (tuple) optional: the RGB values for the desired color (3 integers) (default None)
        * font_pos (tuple) optional: the (x,y) coordinates to anchor the text (2 integers) (default (0, 0))
        * text_box () optional:
        * display (bool) optional: whether to display the tiled images
        * save (str) optional: the filename or path to save the tiled images to. Use None to not save (default None)
        * border (int) optional: the size of border to add to each image (default 20)
        * cmap (str) optional: the colormap for matplotlib to use to map scalar data to colors (default "viridis")
        * quality (int) optional: the image quality to save with. Minimum (worst) is 0, maximum (best) is 90 (default 75)

    returns:
        * tiled ():
    """
    if not isinstance(images, list):
        print("expected list of images")
        return
    
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
    """DESCRIPTION

    params:
        * img ():
        * size (tuple) optional: the integer width and height values (default (16,9))
        * save (str) optional: the filename or path to save the tiled images to. Use None to not save (default None)
    """
    if isinstance(img, list):
        img = tile_imgs(img, save=save)

    plt.figure(figsize=size)
    plt.imshow(img)
    plt.axis('off')


def match_scale(pred, grnd, mask=None, skip_close=False, threshold=0.001, subsample=1.0):
    """DESCRIPTION

    params:
        * pred ():
        * grnd ():
        * mask () optional:
        * skip_close (bool) optional:
        * threshold (float) optional:
        * subsample (float) optional:

    returns:
        * ():
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
    """DESCRIPTION

    params:
        * a ():
        * b ():
        * mask () optional:
        * subsample (float) optional:

    returns:
        * scale ():
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


def get_brightness(rgb, mode='numpy'):
    """DESCRIPTION

    params:
        * rgb ():
        * mode (str) optional:

    returns:
        * brightness ():
    """
    # "CCIR601 YIQ" method for computing brightness
    if mode == 'numpy':
        brightness = (0.3 * rgb[:,:,0]) + (0.59 * rgb[:,:,1]) + (0.11 * rgb[:,:,2])
        return brightness[:, :, np.newaxis]
    if mode == 'torch':
        brightness = (0.3 * rgb[0,:,:]) + (0.59 * rgb[1,:,:]) + (0.11 * rgb[2, :,:])
        return brightness.unsqueeze(0)


def minmax(img):
    """DESCRIPTION

    params:
        * img ():

    returns:
        * img ():
    """
    return (img - img.min()) / img.max()


def inv_2_real(inv_shd):
    """DESCRIPTION

    params:
        * inv_shd ():

    returns:
        * shd ():
    """
    # first normalize the network inverse shading to be [0,1]
    norm_inv_shd = minmax(inv_shd)

    # convert to regular shading, clip very small values for division
    shd = (1.0 / norm_inv_shd.clip(1e-5)) - 1.0

    return shd.clip(1e-5)


def round_32(x):
    """Round a number up to the next multiple of 32.

    params:
        * x (numeric): a number to round

    returns:
        * (int): x rounded up to the next multiple of 32
    """
    return 32 * math.ceil(x / 32)


def to2np(img):
    """Convert a torch image with dimensions (channel, height, width) to a
    numpy image with dimensions (height, width, channel).

    params:
        * img (torch.Tensor): a torch image with dimensions (c, h, w)

    returns:
        * img (numpy.array): the image converted to numpy (h, w, c)
    """
    return img.detach().cpu().permute(1, 2, 0).numpy()


def view_scale(img, p=100):
    """DESCRIPTION

    params:
        * img ():
        * p (int) optional:

    returns:
        * img ():
    """
    return (img / np.percentile(img, p)).clip(0, 1)


def view(img, p=100):
    """DESCRIPTION

    params:
        * img ():
        * p (int) optional:

    returns:
        * img ():
    """
    return view_scale(img ** (1/2.2), p=p)


def invert(x):
    """DESCRIPTION

    params:
        * x ():

    returns:
        * out ():
    """
    out = 1.0 / (x + 1.0)
    return out


def uninvert(x, eps=0.001, clip=True):
    """DESCRIPTION

    params:
        * x ():
        * eps (float) optional:
        * clip (bool) optional:

    returns:
        * out ():
    """
    if clip:
        x = x.clip(eps, 1.0)

    out = (1.0 / x) - 1.0
    return out


def get_tonemap_scale(rgb_color, p=90):
    """DESCRIPTION

    params:
        * rgb_color ():
        * p (int) optional:

    returns:
        * scale ():
    """
    gamma                             = 1.0 / 2.2   # standard gamma correction exponent
    inv_gamma                         = 1.0 / gamma
    # percentile                        = 90        # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8       # ...to be this bright after scaling

    brightness       = get_brightness(rgb_color)
    # brightness_valid = brightness[valid_mask]

    eps                               = 0.0001 # if the kth percentile brightness value in the unmodified image is less than this, set the scale to 0.0 to avoid divide-by-zero
    brightness_nth_percentile_current = np.percentile(brightness, p)

    if brightness_nth_percentile_current < eps:
        scale = 0.0
    else:
        # Snavely uses the following expression in the code at https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
        # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma - np.log(brightness_nth_percentile_current))
        #
        # Our expression below is equivalent, but is more intuitive, because it follows more directly from the expression:
        # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired
        scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

    return scale
