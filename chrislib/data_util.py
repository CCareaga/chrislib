import os
import random
import urllib
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from bs4 import BeautifulSoup


def load_from_url(url):
    """Load an image into a numpy.array from a URL.

    params:
        url (str): the URL that points to the image to fetch

    returns:
        (numpy.array): the image loaded as a numpy float array
    """
    response = requests.get(url, timeout=60)
    return load_image(BytesIO(response.content))


def load_image(path, bits=8):
    """Load an image into a numpy.array from a filepath or file object.

    params:
        path (str or file): the filepath to open or file object to load
        bits (int) optional: bit-depth of the image for normalizing to [0-1] (default 8)

    returns:
        (numpy.array): the image loaded as a numpy array
    """
    np_arr = np.array(Image.open(path)).astype(np.single)
    return np_arr / float((2 ** bits) - 1)


def load_depth(path, bit_depth=16):
    """Load a depth map into a numpy.array from a filepath or file object.

    params:
        path (str or file): the filepath to open or file object to load
        bits (int) optional: bit-depth of the depth map (default 16)

    returns:
        (numpy.array): the depth map loaded as a numpy array
    """
    depth = Image.open(path)
    depth_arr = np.array(depth).astype(np.single)
    depth_arr = depth_arr / (2**bit_depth)

    return depth_arr.astype(np.single)


def np_to_pil(img, bits=8):
    """Convert a [0-1] numpy array into a PIL image.

    params:
        img (numpy.array): the numpy array to convert to PIL Image
        bits (int) optional: desired bit-depth of output PIL Image (default 8)

    returns:
        (PIL.Image): the image converted to a PIL Image object
    """
    if bits == 8:
        int_img = (img * 255).astype(np.uint8)
    if bits == 16:
        int_img = (img * ((2 ** 16) - 1)).astype(np.uint16)

    return Image.fromarray(int_img)


def random_color_shift(image, mean=0.5, std=0.05):
    """Randomly change the color temperature of an image in RGB space while maintaining luminance."""

    # Generate a random color temp and normalize it to maintain luminance
    temperature = np.random.normal(mean, std, size=3)
    temperature = temperature.clip(0.1, 1)
    temperature /= (np.sum(temperature) / 3)
    
    # Multiply each pixel of the image by the color multiplier
    adjusted_image = image * torch.from_numpy(temperature.reshape(-1, 1, 1))
    # adjusted_image = np.clip(adjusted_image, 0, 1)

    return adjusted_image.clip(0, 1).float()

def random_color_jitter(img):
    """perform a random color jitter on an rgb image for augmentation

    params:
        img (PIL.Image or torch.Tensor): image to be color jittered

    returns:
        sat_img (PIL.Image or torch.Tensor): color jittered image
    """
    hue_shft = (random.randint(0, 50) / 50.) - 0.5
    hue_img = TF.adjust_hue(img, hue_shft)

    sat_shft = (random.randint(0, 50) / 50.) + 0.5
    sat_img = TF.adjust_saturation(hue_img, sat_shft)

    r_mul = 1.0 + (random.randint(0, 100) / 250) - 0.2
    b_mul = 1.0 + (random.randint(0, 100) / 250) - 0.2
    sat_img[0, :, :] *= r_mul
    sat_img[2, :, :] *= b_mul

    return sat_img


def random_crop_and_resize(images, output_size=384,  min_crop=128):
    """Randomly (within the given output_size and min_crop constraints) resize and crop a set of
    images.

    params:
        images (list of torch.Tensor): list of images to randomly crop (same crop on all images)
        output_size (int) optional: the height and width to output the image with (default 384)
        min_crop (int) optional: the minimum allowable cropping amount (default 128)

    returns:
        images (list of torch.Tensor): cropped and resized images
    """
    _, h, w = images[0].shape

    max_crop = min(h, w)

    rand_crop = random.randint(min_crop, max_crop)

    rand_top = random.randint(0, h - rand_crop)
    rand_left = random.randint(0, w - rand_crop)

    images = [TF.crop(x, rand_top, rand_left, rand_crop, rand_crop) for x in images]
    images = [TF.resize(x, (output_size, output_size)) for x in images]

    return images


def random_flip(images, p=0.5, mode='h'):
    """Perform a horizontal flip with a certain probability on a set of images

    params:
        images (list of torch.Tensor): list of images to randomly flip
        p (float) optional: probability of flipping (default 0.5)

    returns:
        (list of torch.Tensor): list of randomly flipped images
    """
    if random.random() < p:
        if mode == 'h':
            images = [TF.hflip(x) for x in images]
        elif mode == 'v':
            images = [TF.vflip(x) for x in images]

    return images


def get_main_img(soup):
    """Extract the URL for the main image from a web page's content. This function is specifically
    for downloading images from the OpenSurfaces website

    params:
        soup (BeautifulSoup): the content of a web page loaded with beautifulsoup

    returns:
        link (str): the link to the main image from the soup content
    """
    ret_link = ""
    for entry in soup.findAll('a'):
        link = entry.get('href')
        if link is None:
            continue

        if 'http://labelmaterial.s3.amazonaws.com/photos/' in link:
            ret_link = link
            break
    return ret_link


def load_opensurfaces_image(id_num):
    """Load an image, as specified by the id_num, from opensurfaces.cs.cornell.edu as a numpy array.

    params:
        id_num (int): the ID of the image to load

    returns:
        img_arr (numpy.array): the image loaded as a numpy array
        soup (BeautifulSoup): the page content as a BeautifulSoup object
    """
    IMAGE_PATH = '/home/chris/research/intrinsic/data/opensurfaces/images'

    url = f'http://opensurfaces.cs.cornell.edu/photos/{id_num}/'
    r = requests.get(url, timeout=60)
    content = r.content

    soup = BeautifulSoup(content)

    img_url = get_main_img(soup)

    img_base = img_url.split('/')[-1]
    img_path = os.path.join(IMAGE_PATH, img_base)

    # check if the image is already downloaded
    if not os.path.exists(img_path):
        print("downloading original image file")
        urllib.request.urlretrieve(img_url, img_path)

    img_arr = load_image(img_path)

    h, w, _ = img_arr.shape

    return img_arr, soup
