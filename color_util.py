import numpy as np

from .general import invert, uninvert

def get_brightness(rgb, mode='numpy'):

    # "CCIR601 YIQ" method for computing brightness
    if mode == 'numpy':
        brightness = (0.3 * rgb[:,:,0]) + (0.59 * rgb[:,:,1]) + (0.11 * rgb[:,:,2])
        return brightness[:, :, np.newaxis]
    if mode == 'torch':
        brightness = (0.3 * rgb[0,:,:]) + (0.59 * rgb[1,:,:]) + (0.11 * rgb[2, :,:])
        return brightness.unsqueeze(0)

def rgb2yuv(rgb, clip=True):
    m = np.array([
        [0.299, -0.147,  0.615],
        [0.587, -0.289, -0.515],
        [0.114,  0.436, -0.100]
    ])
    yuv = np.dot(rgb, m)
    yuv[:,:,1:] += 0.5  
    
    if clip:
        yuv = yuv.clip(0, 1)
        
    return yuv.clip(0, 1)

def yuv2rgb(yuv, clip=True):
    m = np.array([
        [1.000,  1.000, 1.000],
        [0.000, -0.394, 2.032],
        [1.140, -0.581, 0.000],
    ])
    yuv[:, :, 1:] -= 0.5
    rgb = np.dot(yuv, m)
    
    if clip:
        rgb = rgb.clip(0, 1)
        
    return rgb

def rgb2luv(rgb, eps=0.001):

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    l = (r * 0.299) + (g * 0.587) + (b * 0.114)
    u = invert(r / (g + eps))
    v = invert(b / (g + eps))

    return np.stack((l, u, v), axis=-1)

def luv2rgb(luv, eps=0.001):

    l = luv[:, :, 0]
    u = uninvert(luv[:, :, 1], eps=eps)
    v = uninvert(luv[:, :, 2], eps=eps)

    g = l / ((u * 0.299) + (v * 0.114) + 0.587)
    r = g * u
    b = g * v

    return np.stack((r, g, b), axis=-1)

def batch_rgb2luv(rgb, eps=0.001):
    r = rgb[:, 0, :, :]
    g = rgb[:, 1, :, :]
    b = rgb[:, 2, :, :]

    l = (r * 0.299) + (g * 0.587) + (b * 0.114)
    u = invert(r / (g + eps))
    v = invert(b / (g + eps))

    return torch.stack((l, u, v), axis=1)

def batch_luv2rgb(luv, eps=0.001):
    l = luv[:, 0, :, :]
    u = uninvert(luv[:, 1, :, :], eps=eps)
    v = uninvert(luv[:, 2, :, :], eps=eps)

    g = l / ((u * 0.299) + (v * 0.114) + 0.587)
    r = g * u
    b = g * v

    return torch.stack((r, g, b), axis=1)
