# NPUBXY solution of the [LLD-MMRI Challenge](https://github.com/LMMMEng/LLD-MMRI2023)


## Preprocessing
- Begin by restoring the original spacing for each NII file using the script  ```misc/remake_original_image.py```.
- Conduct a visual inspection to ensure the accuracy of all images. Utilize the code in  ```misc/visual_image.py```.
I have found some cases in the training set are flipped upside down. These misoriented images were subsequently corrected manually.
- Perform image registration on all eight-phase images. The reference image for registration is the C+pre phase, and all other phases are aligned to this reference. 
I use [Cross-SAM](https://arxiv.org/pdf/2307.03535.pdf) for registration.
- The registered patches, along with our trained model, have been uploaded via [Baidu Disk](https://pan.baidu.com/s/1WTeRtFqzvSmpHtzoqNzHdQ?pwd=299z)
  (pw:299z).



## Solution

- Our approach utilizes a similar framework as the [baseline](https://github.com/LMMMEng/LLD-MMRI2023/tree/main/main). However, we replace the 3D UniFormer with a 2D ResNet18 architecture. 
During training, we randomly select 3 consecutive slices from each cropped case, the output features
for all 8 phases are fused by the final FC layer.
- For inference, we only use the center 3 slices of each case.
- To enhance our model's performance, we employ a two-level model ensemble strategy. For a model trained using a specified train/validation split,
we identify the top five models exhibiting the highest overall F1-score and kappa values (referred to as best_fk_checkpoint).
Additionally, we train models using distinct train/validation splits. In our final solution, we incorporate a total of seven different train/validation splits.

## Data and pre-trained models
- The data_model_labels.zip ([Baidu Disk](https://pan.baidu.com/s/1WTeRtFqzvSmpHtzoqNzHdQ?pwd=299z) pw:299z) contains three folders:
```
data_model_labels
  - images_mycrop_8 (contains all cropped registered patches with margin of 8)
  - labels (training and validation data split)
  - models (all pretrained models)
```

## Inference
- Our training regimen involves the utilization of seven distinct train/validation split models. For each of these models,
we select the top five checkpoints based on their F1 and kappa scores. Consequently, a total of 35 models need to be utilized
for inference. The final result is obtained by calculating the average of predictions from each individual model.
- The predictions for each model have been uploaded to the ```models``` folder, organized as follows:
```
models
  - resnet18m-1
    - NPUBXY_fk_0.json
    - NPUBXY_fk_1.json
    ...
```
- To reproduce our results, you can conveniently execute the script  ```python misc/emsem_result_subtraining.py``` (remember to change the path).
This will automatically generate the final average prediction using all ```NPUBXY_fk_*.json``` files across all seven train/val split models.
The output of this process will precisely match the results submitted during the testing phase.
- Alternatively, you have the option to use the script ```multi_model_predictions_subtraining.py``` to  generate 
predictions for each individual model. It's important to note that we employ test-time random augmentation, which might
lead to slight variations in the results.

## Training
- For training new models with our designated train/validation split, please make use of the code ```multi_model_training.py```.

