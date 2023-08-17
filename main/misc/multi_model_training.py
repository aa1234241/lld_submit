import os
import numpy as np
import subprocess

validate_ratio = 0.2

all_anno_name = 'train_and_val.txt'
anno_path = '/media/userdisk0/code/lld_submit/models/labels/'


run_name = ''
for jj in range(1,8):


    cmd = ['/home/xiaoyubai/anaconda3/envs/sam/bin/python',
           '/media/userdisk0/code/lld_submit/main/train_new.py',
           '--data_dir', '/media/xiaoyubai/raw_data/lld_mmri/data/classification_dataset/images_mycrop_8/',
           '--train_anno_file',
           anno_path + 'train_' + str(
               jj) + '.txt',
           '--val_anno_file',
              anno_path + 'val_' + str(jj) + '.txt',
           '--model', 'resnet18m','--batch-size', '4','--validation-batch-size','1',
           '--num-classes','7',
           '--lr', '1e-4', '--warmup-epochs', '5', '--epochs', '300', '--output',
           '/media/userdisk0/code/lld_submit/output/'
           ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stderr.decode())

print('test')
