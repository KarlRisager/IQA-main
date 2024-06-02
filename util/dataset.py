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
from . import preprocessing



class NiftiDataset(Dataset):
    def __init__(self, folder, file_paths, dimension=0, num_slices=256):
        self.folder = folder
        self.file_paths = file_paths
        self.dimension = dimension
        self.num_slices = num_slices

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = preprocessing.normalize_tensor(torch.tensor(preprocessing.load_nifti(self.folder + self.file_paths[idx])))
        slices = []
        num_slices = image.shape[self.dimension]
        for i in range(num_slices):
            i = int(i)
            if self.dimension == 0:
                slice_image = image[i, :, :]
            elif self.dimension == 1:
                slice_image = image[:, i, :]
            else:  # Assuming dimension 2
                slice_image = image[:, :, i]
            slices.append(slice_image.unsqueeze(0))
        stacked_slices = torch.stack(slices, dim=0)
        return {'images': stacked_slices.repeat(1, 3, 1, 1)}