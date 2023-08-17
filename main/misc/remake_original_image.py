import torchio as tio
import os
import json
import nibabel as nib
import numpy as np

anno_file = open('/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/labels/Annotation_test.json', 'rb')
image_path = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/images/'
save_path = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/images-recover/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
all_annos = json.load(anno_file)
cases_info = all_annos['Annotation_info']
label_name = all_annos['Category_info']

# flipped_train_case_name = ['MR745', 'MR2126', 'MR2603', 'MR3573', 'MR8888', 'MR11115', 'MR11696', 'MR13219', 'MR13467',
#                      'MR13762', 'MR14342', 'MR14400', 'MR17022', 'MR17217', 'MR17289', 'MR19145']
flipped_test_case_name = ['MR1872','MR11656','MR14362','MR14624','MR18858']
for case in cases_info:
    print(case)
    if case not in flipped_test_case_name:
        continue
    # if 'MR2603' not in case:
    #     continue
    if not os.path.exists(os.path.join(save_path, case)):
        os.mkdir(os.path.join(save_path, case))
    single_case_info = cases_info[case]
    case_phases_info = {}
    for phase_info in single_case_info:
        case_phases_info[phase_info['phase']] = {}
        for key in phase_info:
            if 'phase' not in key:
                case_phases_info[phase_info['phase']][key] = phase_info[key]

    for phase in case_phases_info:
        phase_info = case_phases_info[phase]
        img = tio.ScalarImage(
            os.path.join(image_path, case, phase_info['studyUID'], phase_info['seriesUID'] + '.nii.gz'))
        xy_spacing = phase_info['pixel_spacing']
        z_spacing = phase_info['slice_spacing']
        origin = phase_info['origin']
        # if 'C' in phase:
        #     z_spacing = z_spacing/2

        img_data = img.data[0, :, :, :].numpy()
        # img_data = img_data[:, :, ::-1]

        affine_matrix = np.array(
            [[-xy_spacing[0], 0, 0, -origin[0]], [0, -xy_spacing[1], 0, -origin[1]], [0, 0, z_spacing, origin[2]],
             [0, 0, 0, 1]])
        nii_img = nib.Nifti1Image(img_data, affine=affine_matrix)
        nib.save(nii_img, os.path.join(save_path, case, phase + '.nii.gz'))
