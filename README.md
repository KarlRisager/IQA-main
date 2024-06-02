# IQA-main
 
This repository contains the code for the quality assessment framework described in 'QUALITY ASSESSMENT OF BRAIN-IMAGING BASED GENERATIVE AI MODELS'. The code evaluates the quality of synthetic 3D MRI images compared to a real set of 3D MRI images. To compare two sets of 3D images, first install the packages in the `requirements.txt` file, and then run:

```bash
python3 generate_metrics.py --real_folder=<real_folder_path> --generated_folder=<generated_folder_path> --batch_size=1
