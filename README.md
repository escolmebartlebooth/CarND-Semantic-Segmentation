# Udacity Self Driving Car Nanodegree: Term 3: Semantic Segmentation Project

Author: David Escolme

Date of Version: 22 October 2018

Version: Draft


### Objectives

Take a pre-trained convolutional network and implement a fully connected version to do binary classification for semantic segmentation of road images. 

Additionally, apply pre-processing on images to improve network performance, create a pipeline for inference on a video and attempt to create a similar network over N classes.


### Dependencies and Setup

##### GPU

`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.

##### Frameworks and Packages

Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset

Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Code walk through and reflection

#### The data

TO DO: Show GT, Image and discuss the data

#### main.py to create the base FCN for the Kitti Road Dataset

TO DO: Load VGG

TO DO: Layers

TO DO: Optimize

TO DO: TRAIN

TO DO: Loss during training, params {kp, batch, epoch, reg, init, l rate}

#### Updating helper.py to pre-process the Kitti Road Dataset Images


#### Saving and Restoring the model to be run in prediction so a Video pipeline could be created


#### Adapting this approach and applying it to the ... dataset


#### Modes of operation

Run the following command to run the project for basic FCN on Kitti Dataset with image pre-processing:
```
python main.py
```

TO DO: prediction

TO DO: City

### Improvements


 
### Notes from the original Udacity README

- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

