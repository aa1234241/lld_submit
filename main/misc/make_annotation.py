import torchio as tio
import os
import json
import numpy as np

anno_file = open('/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/labels/Annotation_test.json', 'rb')
image_path = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/images-recover-registered/'

all_annos = json.load(anno_file)
cases_info = all_annos['Annotation_info']
label_name = all_annos['Category_info']

registered_anno = {}

flipped_train_case_name = ['MR745', 'MR2126', 'MR2603', 'MR3573', 'MR8888', 'MR11115', 'MR11696', 'MR13219', 'MR13467',
                     'MR13762', 'MR14342', 'MR14400', 'MR17022', 'MR17217', 'MR17289', 'MR19145']
flipped_test_case_name = ['MR1872','MR11656','MR14362','MR14624','MR18858']

for case in cases_info:
    print(case)

    flag = 0
    if case in flipped_test_case_name:
        flag = 1
    single_case_info = cases_info[case]
    case_phases_info = {}
    for phase_info in single_case_info:
        case_phases_info[phase_info['phase']] = {}
        for key in phase_info:
            if 'phase' not in key:
                case_phases_info[phase_info['phase']][key] = phase_info[key]

    for phase in case_phases_info:
        if 'C-pre' not in phase:
            continue
        phase_info = case_phases_info[phase]
        img = tio.ScalarImage(
            os.path.join(image_path, case, phase + '.nii.gz'))
        z_shape = img.shape[3]
        annotations = phase_info['annotation']
        for i in range(annotations['num_targets']):
            lesion_anno = annotations['lesion'][str(i)]
            if flag == 0:
                twoDs = []
                for bbx in lesion_anno['bbox']['2D_box']:
                    bbx['slice_idx'] =z_shape - bbx['slice_idx']
            twoDs.append(lesion_anno['bbox']['2D_box'])
            threeD = lesion_anno['bbox']['3D_box']
            if flag == 0:
                threeD['z_min'] = z_shape - threeD['z_min']
                threeD['z_max'] = z_shape - threeD['z_max']
                tmp = threeD['z_min']
                threeD['z_min'] =  threeD['z_max']
                threeD['z_max'] = tmp
            lesion_anno['bbox']['2D_box'] = twoDs
            lesion_anno['bbox']['3D_box'] = threeD
            annotations['lesion'][str(i)] = lesion_anno
            print('test')
        registered_anno[case] = phase_info['annotation']

f = open('/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/labels/Annotation_mycrop.json', 'w')
json.dump(registered_anno,f)

