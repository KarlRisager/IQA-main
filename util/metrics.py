import torch
import numpy as np
from piq import FID, brisque
import os
import nibabel as nib
from nibabel.orientations import aff2axcodes, apply_orientation, axcodes2ornt, ornt_transform, inv_ornt_aff
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from nibabel.processing import resample_to_output
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import sys




def ssim(image1, image2):
    image1 = image1.cpu().numpy()
    image2 = image2.cpu().numpy()
    data_range = np.max(image1) - np.min(image1)
    return structural_similarity(image1, image2, data_range=data_range, full=True)[0]

def psnr(image1, image2):
    image1 = image1.cpu().numpy()
    image2 = image2.cpu().numpy()
    data_range = np.max(image1) - np.min(image1)
    mse = np.mean((image1 - image2) ** 2)

    if mse == 0:
        return float('inf')
    return peak_signal_noise_ratio(image1, image2, data_range=data_range)

def normalize_image(X):
    X_min = torch.min(X)
    X_max = torch.max(X)
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized

def calculate_brisque(image, dim):
    normalized_image = normalize_image(image)
    brisque_values = []

    for slice_index in range(normalized_image.shape[dim]):
        if dim == 0:
            slice_2d = normalized_image[slice_index, :, :]
        elif dim == 1:
            slice_2d = normalized_image[:, slice_index, :]
        elif dim == 2:
            slice_2d = normalized_image[:, :, slice_index]

        if torch.all(slice_2d == 0.0):
            continue

        if torch.var(slice_2d) < 1e-6: 
            continue


        slice_2d = slice_2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        brisque_value = brisque(slice_2d) 
        brisque_values.append(brisque_value.item())

    mean_brisque_value = np.mean(np.array(brisque_values))
    return mean_brisque_value