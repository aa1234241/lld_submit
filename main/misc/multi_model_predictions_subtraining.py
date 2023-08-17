import os
import cmd
import subprocess

all_exams_path = '/media/userdisk0/code/lld_submit/models/'
all_exams = os.listdir(all_exams_path)
all_exams = [exam for exam in all_exams if os.path.isdir(os.path.join(all_exams_path, exam)) ]

metric = 'fk'
for exam in all_exams:
    contents = os.listdir(os.path.join(all_exams_path, exam))
    f1_model_names = [name for name in contents if 'best_'+metric in name]
    for i,f1_model_name in enumerate(f1_model_names):
        print(f1_model_name)
        cmd = ['/home/xiaoyubai/anaconda3/envs/sam/bin/python','/media/userdisk0/code/lld_submit/main/predict_new.py',
               '--data_dir','/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/classification_dataset/images_mycrop_8/',
               '--val_anno_file', '/media/xiaoyubai/raw_data/lld_mmri/lldmmri_test_set/classification_dataset/labels/labels_test_inaccessible.txt',
               '--model', 'resnet18m','--num-classes','7',
               '--batch-size', '1', '--checkpoint',
               all_exams_path+exam+'/'+f1_model_name, '--results-dir',
               all_exams_path+exam+'/', '--team_name', 'NPUBXY_'+metric+'_'+str(i)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stderr.decode())
print('test')
