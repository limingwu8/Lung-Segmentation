# Lung-Segmentation

## Overview
This is the code for lung segmentation on RSNA pneumonia detection dataset. The whole dataset can be downloaded from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge. In order to remove the unnecessary features from the CT image and only keep the lung area, a U-Net model is implemented to segment the lung out from the CT image. I manually labeled the contour of 1000 CT images and use these images as training set to train a U-Net. The segmented images can be find at https://drive.google.com/drive/folders/1gISKPOiDuZTAXkGeQ6-TMb3190v4Xhyc?usp=sharing. 

## Prerequisites
* Pytorch 3.1, Python 3.6
* tqdm, visdom

## Model
The schematic of the U-Net model I used for this task.
![image1](https://github.com/limingwu8/Lung-Segmentation/blob/master/images/model.png)
A batch of single channel 512x512 images are feed into the network. The feature extraction is performed by a series of CNN layers. The blue arrow represents a CNN block, which is the combination of a convolution layer, batch normalization layer and ReLU layer. The kernel of the convolution layer has the size 3x3, stride 2, and zero padding. The double-arrow denotes the feature concatenation. Finally, a batch of 512x512x1 probability matrix is output to represent the segmented image. The binary cross-entropy loss is calculated between the input image and the output prediction. The Adam optimizer is used with learning rate 1e-3 and weight decay 1e-4. Since the huge amount of parameters in U-Net, the model is parallelized in two Nvidia GTX 1080 graphic cards with 8 images for one batch. 

## Evaluation
![image2](https://github.com/limingwu8/Lung-Segmentation/blob/master/images/loss.png)
Since the purpose of the segmentation is not precisely segment the lung but remove the other unrelated features for better classification, the lung of the 1000 training images are roughly labeled by myself. The total images are divided into 800 images for training and 200 images for validation. The final IoU is around 0.9.
## Demo
![image3](https://github.com/limingwu8/Lung-Segmentation/blob/master/images/demo.png)

