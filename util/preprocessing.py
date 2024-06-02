import argparse
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





def make_RAS(img):
    '''Take nifti file and returns RAS orientet data'''
    axis = aff2axcodes(img.affine)

    if axis == ('R', 'A', 'S'):
        return img.get_fdata()
    
    current_ornt = axcodes2ornt(axis)
    target_ornt = axcodes2ornt(('R','A','S'))
    transform_matrix = ornt_transform(current_ornt, target_ornt)

    img_data = apply_orientation(img.get_fdata(), transform_matrix)
    return img_data

def pad_crop(img, target = (256,256,256)):
    shape = img.shape
    pad_width = np.array([
        ((target[0] - shape[0])//2, (target[0] - shape[0]+1)//2),
        ((target[1] - shape[1])//2, (target[1] - shape[1]+1)//2),
        ((target[2] - shape[2])//2, (target[2] - shape[2]+1)//2)
    ])

    pad_width1 = np.copy(pad_width)
    pad_width[pad_width<0] = 0

    pad_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    shape = pad_img.shape



    pad_width1[pad_width1>0] = 0

    pad_width = pad_width1

    p = [int(shape[0]/2 != shape[0]//2), int(shape[1]/2 != shape[1]//2), int(shape[2]/2 != shape[2]//2)]

    left_outer0 = pad_img[:((shape[0]+1)//2)+pad_width[0][0], :, :]
    right_outer0 = pad_img[(shape[0]//2)-pad_width[0][1]+ p[0]:, :, :]
    cropped_img = np.concatenate([left_outer0, right_outer0], axis=0)


    left_outer1 = cropped_img[:, :((shape[1]+1)//2)+ pad_width[1][0], :]
    right_outer1 = cropped_img[:, (shape[1]//2)-pad_width[1][1]+ p[1]:, :]
    cropped_img = np.concatenate([left_outer1, right_outer1], axis=1)



    left_outer2 = cropped_img[:, :, :((shape[2]+1)//2)+ pad_width[2][0]]
    right_outer2 = cropped_img[:, :, (shape[2]//2)- pad_width[2][1]+ p[2]:]
    cropped_img = np.concatenate([left_outer2, right_outer2], axis=2)

    return cropped_img



def load_nifti(file_path):
    import nibabel as nib
    img = nib.load(file_path)
    hdr = img.header
    
    voxel_dimensions = hdr.get_zooms()
    target_voxel_dimensions = (1.0, 1.0, 1.0)

    img = resample_to_output(img, target_voxel_dimensions)
    img = make_RAS(img)
    img = pad_crop(img)
    return img

def normalize_tensor(tensor):
    max_val = tensor.max()
    min_val = tensor.min()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
