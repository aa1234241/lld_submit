# Xiaoyu Bai, NPWU
# some images have error added 1024 on some slices
# ckeck the images on their empty ( background) region
# if their intensity is larger than 1000, minus it

import os
import torchio as tio
import numpy as np
import nibabel as nib

file_root_path = '/media/xiaoyubai/raw_data/lld_mmri/data/images-recover/'
case_name = 'MR58043'

all_phases_name = os.listdir(os.path.join(file_root_path,case_name))

for phase_name in all_phases_name:
    phase_img = tio.ScalarImage(os.path.join(file_root_path,case_name,phase_name))
    phase_img_np = phase_img.data.numpy()[0,:,:,:]
    for s in range(phase_img_np.shape[2]):
        v = phase_img_np[10,10,s]
        if v >200:
            phase_img_np[:, :, s] = phase_img_np[:, :, s] - v
    phase_affine = phase_img.affine
    nii_img = nib.Nifti1Image(phase_img_np, affine=phase_affine)
    nib.save(nii_img, os.path.join(file_root_path,case_name,phase_name))
    print('test')
