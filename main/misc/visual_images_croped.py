import os
import numpy as np
import matplotlib.pyplot as plt
import torchio as tio
import json

f = open('/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/labels/Annotation_mycrop.json', 'r')
annos = json.load(f)

all_cases_path = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/classification_dataset/images_mycrop_8/'
all_cases = os.listdir(all_cases_path)
save_path = '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/visual-cropped_8/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# label_path = '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/labels/'
# all_validate_cases = np.loadtxt(os.path.join(label_path,'random_split_val_10.txt'), dtype=np.str_)

for caseinfo in all_cases:
    case = caseinfo
    if '111774' not in case:
        continue
    # anno = annos[case]
    # label = anno['lesion']['0']['category']
    phase_images = os.listdir(os.path.join(all_cases_path,case))
    all_phase_name = [name.split('.',1)[0] for name in phase_images]
    image_lists = []
    image_spacings = []
    for image_name in phase_images:
        image = tio.ScalarImage(os.path.join(all_cases_path,case,image_name))
        image_numpy = image.data[0,:,:,:].numpy()
        image_spacing = image.spacing
        image_lists.append(image_numpy)
        image_spacings.append(image_spacing)
    fig, ax = plt.subplots(2, 4, figsize=(40, 20))
    i = 0
    for name,image,spacing in zip(all_phase_name,image_lists,image_spacings):
        ax[i//4, i%4].set_title(name)
        image_slice = image[:,:,image.shape[2]//2]
        # image_slice = image_slice[:,::-1]
        # image_slice = np.transpose(image_slice,(1,0))
        ax[i//4, i%4].imshow(image_slice, cmap='gray', aspect=spacing[2] / spacing[1])
        i = i+1
    plt.savefig(
        f'{save_path}/{case}_{-1}.png')
    plt.close()
    print('test')
