import os
import sys
import random
from PIL import Image
import numpy as np
import cv2
import skimage
import torchvision.transforms.functional as TF
import requests
import urllib
from io import BytesIO
from bs4 import BeautifulSoup

def load_from_url(url):
    response = requests.get(url)
    return load_image(BytesIO(response.content))

def load_image(path):
    return np.array(Image.open(path)).astype(np.float32) / 255.

def load_depth(path, bit_depth=16):
    depth = Image.open(path)
    depth_arr = np.array(depth).astype(np.float32)
    depth_arr = depth_arr / (2**bit_depth)

    return depth_arr.astype(np.float32)

def np_to_pil(img):
    """ converts a [0-1] numpy array into a PIL image """
    int_img = (img * 255).astype(np.uint8)
    return Image.fromarray(int_img)

def random_color_jitter(img):

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
    _, h, w = images[0].shape
    
    max_crop = min(h, w)
    
    rand_crop = random.randint(min_crop, max_crop)
    
    rand_top = random.randint(0, h - rand_crop)
    rand_left = random.randint(0, w - rand_crop)
    
    images = [TF.crop(x, rand_top, rand_left, rand_crop, rand_crop) for x in images]
    images = [TF.resize(x, (output_size, output_size)) for x in images]

    return images

def random_flip(images, p=0.5):
    if random.random() > p:
        return [TF.hflip(x) for x in images]
    else:
        return images

def get_main_img(soup):
    for entry in soup.findAll('a'):
        link = entry.get('href')
        if link is None: continue

        if 'http://labelmaterial.s3.amazonaws.com/photos/' in link:
            return link

def load_opensurfaces_image(id_num):
    
    IMAGE_PATH = '/home/chris/research/intrinsic/data/opensurfaces/images'

    url = f'http://opensurfaces.cs.cornell.edu/photos/{id_num}/'
    r = requests.get(url)
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
    print(h, w)
    
    return img_arr, soup
