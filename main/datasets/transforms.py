import random
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from scipy import ndimage
from timm.models.layers import to_3tuple


def load_nii_file(nii_image):
    image = sitk.ReadImage(nii_image)
    image_array = sitk.GetArrayFromImage(image)
    return image_array


def resize3D(image, size):
    size = to_3tuple(size)
    image = image.astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    x = F.interpolate(image, size=size, mode='trilinear', align_corners=True).squeeze(0)
    x = x.cpu().numpy()
    for i in range(x.shape[0]):
        x[i, :, :, :] = image_normalization(x[i, :, :, :])
    return x

def resize3D_mod(image, size):
    size = to_3tuple(size)
    size_use = np.ceil(np.array(size) * np.random.uniform(0.8, 1.2)).astype(int)
    size_use_tuple = tuple(size_use)
    image = image.astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    x = F.interpolate(image, size=size_use_tuple, mode='trilinear', align_corners=True).squeeze(0).squeeze(0)

    empty_size = (np.array(size) * 1.4).astype(int)
    im_empty = np.zeros([x.shape[0], empty_size[0], empty_size[1], empty_size[2]], dtype=np.float32)

    margin = np.floor((empty_size - size_use) / 2).astype(int)
    for i in range(x.shape[0]):
        im_empty[i, margin[0]:margin[0] + size_use[0], margin[1]:margin[1] + size_use[1],
        margin[2]:margin[2] + size_use[2]] = image_normalization(x[i, :, :, :].cpu().numpy())
    return im_empty


def image_normalization(image, win=None, adaptive=True):
    if win is not None:
        image = 1. * (image - win[0]) / (win[1] - win[0])
        image[image < 0] = 0.
        image[image > 1] = 1.
        return image
    elif adaptive:
        min, max = np.min(image), np.max(image)
        image = (image - min) / (max - min)
        return image
    else:
        return image



def random_crop(image, crop_shape):
    crop_shape = to_3tuple(crop_shape)
    _, z_shape, y_shape, x_shape = image.shape
    z_min = np.random.randint(0, z_shape - crop_shape[0])
    y_min = np.random.randint(0, y_shape - crop_shape[1])
    x_min = np.random.randint(0, x_shape - crop_shape[2])
    image = image[..., z_min:z_min + crop_shape[0], y_min:y_min + crop_shape[1], x_min:x_min + crop_shape[2]]
    return image


def center_crop(image, target_shape=(16, 128, 128)):
    target_shape = to_3tuple(target_shape)
    b, z_shape, y_shape, x_shape = image.shape
    z_min = z_shape // 2 - target_shape[0] // 2
    y_min = y_shape // 2 - target_shape[1] // 2
    x_min = x_shape // 2 - target_shape[2] // 2
    image = image[:, z_min:z_min + target_shape[0], y_min:y_min + target_shape[1], x_min:x_min + target_shape[2]]
    return image


def randomflip_z(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, ::-1, ...]


def randomflip_x(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[..., ::-1]


def randomflip_y(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, :, ::-1, ...]


def random_flip(image, mode='x', p=0.5):
    if mode == 'x':
        image = randomflip_x(image, p=p)
    elif mode == 'y':
        image = randomflip_y(image, p=p)
    elif mode == 'z':
        image = randomflip_z(image, p=p)
    else:
        raise NotImplementedError(f'Unknown flip mode ({mode})')
    return image


def rotate(image, angle=10):
    angle = random.randint(-10, 10)
    r_image = ndimage.rotate(image, angle=angle, axes=(-2, -1), reshape=True)
    if r_image.shape != image.shape:
        r_image = center_crop(r_image, target_shape=image.shape[1:])
    return r_image
