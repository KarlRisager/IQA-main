#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH -p gpu --gres=gpu:h100:1

#SBATCH --job-name=test_quality_synth_150
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=4 


#SBATCH --time=3-00:00:00
#3-00:00:00
#Skipping many options! see man sbatch 
# From here on, we can start our program


python3.11 -m venv IQA_env

# Activate the virtual environment
source IQA_env/bin/activate

pip install -r requirements.txt


python3 generate_metrics.py --real_folder=/home/tzh269/Dir/input_data/synthrad/test/ --generated_folder=/home/tzh269/Dir/wdm-3d-main/results/May22_21-59-32_hendrixgpu05fl.unicph.domain/brats_ours_unet_256_150000/ --batch_size=1
