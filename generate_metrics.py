
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
import gc
from util import preprocessing
from util import dataset
from util import metrics

print('Starting quality assesment')
sys.stdout.flush()
print('--------------------------------------------------------------------------')
sys.stdout.flush()

parser = argparse.ArgumentParser()
parser.add_argument( '--real_folder', type=str, default="real_images/")
parser.add_argument( '--generated_folder', type=str, default="gen_images/")
parser.add_argument( '--batch_size', type=int, default=2)
parser.add_argument( '--run_dimension', type=int, default=2)
parser.add_argument('--test_name', type=str, default='quality_assesment')
parser.add_argument('--generate_real_feats', type=bool, default=False)#If real feats have already been generated it will be loaded from saved file unless this is set to True

args = parser.parse_args()

real_folder = args.real_folder
generated_folder = args.generated_folder
batch_size = args.batch_size
run_dim = args.run_dimension
test_name = args.test_name
generate_real_feats = args.generate_real_feats
print('Getting file paths')
sys.stdout.flush()
#If you wish to only use a certain subset of the test folder, you can index the real_file_paths list accordingly
real_file_paths = os.listdir(real_folder)#[:126]
gen_file_paths = os.listdir(generated_folder)#[:126]





if True: 


    print('Generating dataset')
    sys.stdout.flush()

    real_dataset = dataset.NiftiDataset(real_folder, real_file_paths, dimension=run_dim)
    gen_dataset = dataset.NiftiDataset(generated_folder, gen_file_paths, dimension=run_dim)

    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False)


    print('Computing FID metric')
    sys.stdout.flush()

    fid_metric = FID()
    vgg16_model = models.vgg16(pretrained=True, progress=True).cuda()
    vgg16_model.eval()
    feature_extractor = torch.nn.Sequential(
    vgg16_model.features,
    vgg16_model.avgpool,  
    torch.nn.Flatten(),    
    vgg16_model.classifier[0],  

    )
    len_real_data = len(real_dataset)
    len_gen_data = len(gen_dataset)
    real_feats_file_name = test_name+'_'+'real_feats.pt'
    fid_scores_csv_name = test_name+'_fid_scores.csv'
    real_feats = []



    if not(os.path.exists(real_feats_file_name)) or generate_real_feats:
        print(f'Starting feature extraction of real samples from {real_folder}')
        sys.stdout.flush()
        with torch.no_grad():
            real_feats = []
            for i, batch in enumerate(real_dataloader):
                print(f'process: {i+1} out of {len_real_data} images processed')
                sys.stdout.flush()
                image = batch['images'].squeeze(0).float().cuda()
                N = image.shape[0]
                features = feature_extractor(image)
                features = torch.mean(features, dim=0, keepdim=True)
                real_feats.append(features.cpu())
                del image, features 

        real_feats = torch.cat(real_feats, dim=0)

        print(f'Saving real feats to {real_feats_file_name}')
        sys.stdout.flush()
        torch.save(real_feats, real_feats_file_name)
    else:
        print(f'Loading real feats from file {real_feats_file_name}')
        sys.stdout.flush()
        real_feats = torch.load(real_feats_file_name)


    fid_scores = []
    gen_feats = []
    print(f'Starting feature extraction from generated samples from {generated_folder}')
    sys.stdout.flush()


    with torch.no_grad():
        for i, batch in enumerate(gen_dataloader):
            print(f'process: {i+1} out of {len_gen_data} images processed')
            sys.stdout.flush()
            image = batch['images'].squeeze(0).float().cuda()
            N = image.shape[0]
            features = feature_extractor(image).cpu() 
            features = torch.mean(features, dim=0, keepdim=True)
            gen_feats.append(features)
            del image, features
            torch.cuda.empty_cache()

    gen_feats = torch.cat(gen_feats, dim = 0)

    print(f'Shape of features of real data = {real_feats.shape}')
    print(f'Shape of features of generated data = {gen_feats.shape}')

    total_fid = fid_metric(real_feats, gen_feats)


 
    
    print(f'fid score: {total_fid.item()}')
    sys.stdout.flush()


    del feature_extractor, vgg16_model
    torch.cuda.empty_cache()





ssim_csv_name = test_name+'_ssim.csv'
psnr_csv_name = test_name+'_psnr.csv'
brisque_csv_name = test_name+'_brisque.csv'

ssim_scores = []
psnr_scores = []
brisque_scores = []
len_gen_data = len(gen_file_paths)
print('Computing ssim, psnr and brisque scores of generated images')
sys.stdout.flush()
for i, file in enumerate(gen_file_paths):
    print(f'process: {i+1} out of {len_gen_data} images processed')
    sys.stdout.flush()
    file_path = generated_folder+file
    image1 = torch.tensor(preprocessing.load_nifti(file_path))
    ssim_score = []
    psnr_score = []
    brisque_score = metrics.calculate_brisque(image1, dim=run_dim)
    brisque_scores.append(brisque_score) 

    for j, test_file in enumerate(real_file_paths):
        print(f'Image process {j/len(real_file_paths)}')
        sys.stdout.flush()
        test_file_path = real_folder+test_file
        image2 = torch.tensor(preprocessing.load_nifti(test_file_path))
        ssim_score.append(metrics.ssim(image1, image2))
        psnr_score.append(metrics.psnr(image1, image2))
    ssim_scores.append(np.max(ssim_score))
    psnr_scores.append(np.max(psnr_score))


    np.savetxt(ssim_csv_name, np.array(ssim_scores), delimiter=',')
    np.savetxt(psnr_csv_name, np.array(psnr_scores), delimiter=',')
    np.savetxt(brisque_csv_name, np.array(brisque_scores), delimiter=',')

    del ssim_score, psnr_score, brisque_score, image1, image2
    gc.collect()





print('Quality assesment done')
print('ssim scores')
print('\n')
print(ssim_scores)
print('\n')
print(f'average ssim score: {np.mean(ssim_scores)}')
print('psnr scores')
print('\n')
print(psnr_scores)
print('\n')
print(f'average psnr score: {np.mean(psnr_scores)}')
print('BRISQUE scores\n')
print(brisque_scores)
print('\n')
print(f'average BRISQUE score: {np.mean(brisque_scores)}')
print(f'fid score: {total_fid.item()}')
print('\n')



print(torch.cuda.memory_summary())
print('DONE')
sys.stdout.flush()