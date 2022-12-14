import numpy as np

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
