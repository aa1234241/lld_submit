import os
import sys
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def crop_lesion(data_dir, json_path, save_dir, xy_extension=8, z_extension=2):
    '''
    Args:
        data_dir: path to original dataset
        json_path: path to annotation file
        save_dir: save_dir of classification dataset
        xy_extension: spatail extension when cropping lesion ROI
        z_extension: slice extension when cropping lesion ROI
    '''
    f = open(json_path, 'r')
    data = json.load(f)
    phases = ['C-pre','C+A','C+V','C+Delay','DWI','T2WI','In Phase','Out Phase']

    for patientID in tqdm(data):
        if '111774' not in patientID:
            continue
        for phase in phases:
            annotation = data[patientID]['lesion']

            image_path = os.path.join(data_dir, patientID, phase + '.nii.gz')
            try:
                image = sitk.ReadImage(image_path)
            except KeyboardInterrupt:
                exit()
            except:
                print(sys.exc_info())
                print('Countine Processing')
                continue

            image_array = sitk.GetArrayFromImage(image)

            for ann_idx in annotation:
                ann = annotation[ann_idx]
                lesion_cls = ann['category']
                bbox_info = ann['bbox']['3D_box']

                x_min = int(bbox_info['x_min'])
                y_min = int(bbox_info['y_min'])
                x_max = int(bbox_info['x_max'])
                y_max = int(bbox_info['y_max'])
                z_min = int(bbox_info['z_min'])
                z_max = int(bbox_info['z_max'])
                # bbox = (x_min, y_min, z_min, x_max, y_max, z_max)

                temp_image = image_array

                if z_min >= temp_image.shape[0]:
                    print(f"{patientID}: z_min'{z_min}'>num slices'{temp_image.shape[0]}'")
                    continue
                elif z_max >= temp_image.shape[0]:
                    print(f"{patientID}: z_max'{z_max}'>num slices'{temp_image.shape[0]}'")
                    continue

                if xy_extension is not None:
                    x_padding_min = int(abs(x_min - xy_extension)) if x_min - xy_extension < 0 else 0
                    y_padding_min = int(abs(y_min - xy_extension)) if y_min - xy_extension < 0 else 0
                    x_padding_max = int(abs(x_max + xy_extension - temp_image.shape[1]))if x_max + xy_extension > temp_image.shape[1] else 0
                    y_padding_max = int(abs(y_max + xy_extension - temp_image.shape[2])) if y_max + xy_extension > temp_image.shape[2] else 0

                    x_min = max(x_min - xy_extension, 0)
                    y_min = max(y_min - xy_extension, 0)
                    x_max = min(x_max + xy_extension, temp_image.shape[1])
                    y_max = min(y_max + xy_extension, temp_image.shape[2])
                if z_extension is not None:
                    z_min = max(z_min - z_extension, 0)
                    z_max = min(z_max + z_extension, temp_image.shape[0])

                if temp_image.shape[0] == 1:
                    roi = temp_image[0, y_min:y_max, x_min:x_max]
                    roi = np.expand_dims(roi, axis=0)
                elif z_min == z_max:
                    roi = temp_image[z_min, y_min:y_max, x_min:x_max]
                    roi = np.expand_dims(roi, axis=0)
                else:
                    roi = temp_image[z_min:(z_max+1), y_min:y_max, x_min:x_max]

                if xy_extension is not None:
                    roi = np.pad(roi, ((0, 0), (y_padding_min, y_padding_max), (x_padding_min, x_padding_max)), 'constant')

                nii_file = sitk.GetImageFromArray(roi)
                if int(ann_idx) == 0:
                    save_folder = os.path.join(save_dir, f'{patientID}')
                else:
                    save_folder = os.path.join(save_dir, f'{patientID}_{ann_idx}')
                os.makedirs(save_folder, exist_ok=True)
                sitk.WriteImage(nii_file, save_folder + f'/{phase}.nii.gz')

if __name__ == "__main__":
    # import argparse
    # config_parser = parser = argparse.ArgumentParser(description='Data preprocessing Config', add_help=False)
    # parser = parser.add_argument('--data-dir', default='', type=str)
    # parser = parser.add_argument('--anno-path', default='', type=str)
    # parser = parser.add_argument('--save-dir', default='', type=str)
    # args = parser.parse_args()
    data_dir = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/images-recover-registered/'
    anno_path = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/labels/Annotation_mycrop.json'
    save_dir = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/classification_dataset/images_mycrop_8/'
    # crop_lesion(args.data_dir, args.anno_path, args.save_dir)
    crop_lesion(data_dir, anno_path, save_dir)