# DeepSFM - TensorFlow

## UNDER CONSTRUCTION!

This is a TensorFlow implementation of DeepSFM. This program uses existing depth and pose information to perform bundle adjustment via deep learning.

The network architechture roughly consists of two heads that build depth cost and pose cost volumes. The depth cost volume takes in features extracted from a depth map and its corresponding image using a traditional 2D convolutional feature extractor and samples a hypothesis depth plane. These feature maps are concatenated to form a cost volume. Similarly, the pose cost volume takes the above however now it samples hypothesis rotation (SO(3) group sampling) and translation spaces. Both the depth and depth pose volumes are passed through 3D convolutional layers to extract the final depth map in (L x H x W) and (R, t), respectively. The output of the depth convolutional network is passed through an autoencoder for denoising and context aware refinement.

### Network Architecture

![network architecture](https://github.com/patel-nisarg/DeepSFM-TensorFlow/blob/main/imgs/deepsfm_architecture.png)

Sources:

DeepSFM Paper:
https://arxiv.org/pdf/1912.09697.pdf

Origional Repo:
https://github.com/weixk2015/DeepSFM

Based off of DPSNet:
https://github.com/sunghoonim/DPSNet
