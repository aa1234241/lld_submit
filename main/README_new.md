# My settings of the LLD-MMRI Challenge


## Preprocessing
- First recover the original spacing of each nii file by ```remake_original_image.py```.
- Visually check if all images are correct. Using the ```visual_image.py``` code.
Since I have found some cases in the training set are flipped upside down.



## results
- Using ```multi_model_training.py``` with ```my_crop_4``` data.
- ```my_crop_4``` means the registered data (using Cross-SAM) and cropped the data
by their 3d bounding box annotation on pre phase with a margin of 4. Cropping using the 
```crop_roi_registered.py```.
- Inference using ```my_crop_8```data give the best result.

#### ```crop_4``` training ensemble_20 : output_new_2
| validation data | acc   |f1    | kappa |
|-----------------|-------|-------|-------|
|     crop_8      | 0.8076| 0.7830 |0.7617|
|     crop_4      | 0.7948| 0.7653| 0.7453|
|     crop_16     | 0.7692 |0.7372 |0.7182|


#### ```crop_8``` training ensemble_20 : output_new_1
| validation data  | acc   |f1    | kappa |
|-----------|-------|------|------|
| crop_8    | 0.8076| 0.7713 |0.7635|
| crop_4    | 0.7692| 0.7252| 0.7126|`
| crop_16   | 0.7692 |0.7404|0.7173|