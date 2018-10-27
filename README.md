# Udacity Self Driving Car Nanodegree: Term 3: Semantic Segmentation Project

Author: David Escolme

Date of Version: 27 October 2018

Version Log:

* 22 October: Draft
* 27 October: Exploration and Code Walk through


### Objectives

Take a pre-trained convolutional network and implement a fully connected version to do binary classification for semantic segmentation of road images. 

Optionally, apply pre-processing on images to improve network performance, create a pipeline for inference on a video and attempt to create a similar network over N classes.


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

#### The data and the model

The kitti road dataset has 289 training images. As can be seen below from 1 example, the images are accompanied by a ground truth mapping of pixels to Road (Pink), Not Road (Red) and side road (black)

![Base Image](/images/um_lane_000000.png)
![Labelled Image](/images/um_000000.png)
![Overlay](/images/overlay.png)

The dataset has a test set of images as well.

The aim of semantic segmentation using a fully connected network architecture is to classify each pixel on the image as either 'road' or 'not road'.

The architecture is taken from this paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf


#### main.py to create the base FCN for the Kitti Road Dataset

Parameters for batch size, epochs, drop out, learning rate and other hyper-parameters are maintained at the top of the file. Then a number of functions are declared which build up the model and tensorflow operations for training and prediction. Another file (helper.py) contains functions to prepare and transform data as well as generate batches for training and test images for prediction.

* Load VGG: 
	* This function loads the input, keep probability and 3 intermediate layers of the pre-trained VGG16 model. 
	* These layers are then passed to the layers function to build up our model architecture.
* Layers: 
	* This function creates the fully connected model for training as per the architecture outlined in the paper referenced above. 
	* First, the 3 intermediate layers are passed through a 2d convolution (layers 3 and 4 are scaled as per the notes contained here:)
	* Each of the convolutions has l2 regularisation applied to them
	* Then, the 3 layers go through a pattern of transpose 2d convolutions (upsampling) and addition to one of the other layers (skip layers)
	* Each of the transpose layers also has l2 regularisation applied
	* The resulting model is returned from the function
* Optimize:
	* The optimise function takes the output of the last layer, the correct label tensor, the learning rate and the number of classes
	* the output layer and correct label tensors are trasnformed to 2D (although this should not be strictly necessary?)
	* regularisation losses are calculated as these have to be added to the training loss to make regularisation effective
	* a cross entropy loss is calculated and the final step is to use an Adam optimiser to minimise the loss
* Train
	* The train function generates batches of images for a number of epochs and passes them to the training operation
	* the resulting loss is printed at each epoch
	* the feed dictionary contains the input image tensor, keep probability (drop out), learning rate (used in the adam optimiser) and target images
* Run
	* the preceeding functions create the framework / graph that will be used by tensorflow to train the model
	* the pre-trained VGG16 model is downloaded if necessary
	* the batch image generator is created
	* the graph for tensorflow is initialised by creating placeholder variables and referencing the functions created above
	* the session is run with a saver so the final model can be saved for use in prediction
	* after a full run, the model is saved and test images are passed through the model and saved to a runs/ folder

#### A note on regularisation

The project walk through recommended the use of l2 regularisation. In the first instance, the model above was created without it to see what the output might be. From the image below, it can be seen that the model performs poorly where no regularisation is applied.

![No regularisation](/images/bad_reg.png)


#### Initial results

This base model (with regularisation) was then tested using various combinations of the hyper-parameters. After a limited amount of tuning, the best model loss achieved was 0.128 with 50 epochs, drop out of 0.5, a batch size of 8 and a learning rate of 5E-5.

Example output images can be seen below, showing reasonable if not perfect performance:

![overlay testing](images/test1.png)


#### Updating helper.py to pre-process the Kitti Road Dataset Images

The first optional task was to use data augmentation techniques to improve network performance. This could have taken the form of adding transformed training images to broaden the training set, which was quite small.

I chose to simply apply image normalisation to each training image (and prediction image). This can be seen in helper.py (although it is commented out in this version).

Using this approach, training loss was reduced to 0.106. Further work would be needed here to improve performance by adding transformed images to the data set using clipping, rotation, random image enhancements such as brightness and contrast.

![overlay aug]{images/test2.png)

#### Saving and Restoring the model to be run in prediction so a Video pipeline could be created

The second optional task was to create a pipeline for prediction on a video. This is implemented in prediction.py. Simply, the video is opened using moviepy and a function is passed to the movie (fl_image) which applies the pipeline to each frame before moviepy creates a new video from the altered frames output from the function.

The function opens the saved model from the training run. It captures the logits output, image tensor and keep_probability variable. Each frame is resized to the expected model image size, pre-processed inline with training (see normalisation above) and then run through the model to capture the logits output.

This output is then used as the basis of the new frame.

The output can be see here: [video]{/images/windy_road_output.mp4)

#### Adapting this approach and applying it to the cityscapes dataset




#### Modes of operation

Run the following command to run the project for basic FCN on Kitti Dataset with image pre-processing:
```
python main.py

```
run prediction.py too process the video output or a single image (you will need to review the code to understand the correct paths for input / output)

### Improvements


 
### Notes from the original Udacity README

- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

