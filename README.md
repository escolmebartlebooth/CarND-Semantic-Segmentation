# Udacity Self Driving Car Nanodegree: Term 3: Semantic Segmentation Project

Author: David Escolme

Date of Version: 07 November 2018

Version Log:

* 22 October: Draft
* 27 October: Exploration and Code Walk through
* 28 October: Added references
* 07 November: Finalised for 1st submission


### Objectives

Take a pre-trained convolutional network and implement a fully convolutional version to do binary classification for semantic segmentation of road images.

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

Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training and test images.

### Code walk through and reflection

#### The data and the model

The kitti road dataset has 289 training images. As can be seen below from 1 example, the images are accompanied by a ground truth mapping of pixels to Road (Pink), Not Road (Red) and side road (black)

![Base Image](/images/writeup/um_000000.png)
![Labelled Image](/images/writeup/um_lane_000000.png)
![Overlay](/images/writeup/overlay.png)

The dataset has a test set of images as well.

The aim of semantic segmentation using a fully convolutional network (FCN) architecture is to classify each pixel on the image as either 'road' or 'not road'.

The architecture is taken from this paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

#### main.py to create the base FCN for the Kitti Road Dataset

Parameters for batch size, epochs, drop out, learning rate and other hyper-parameters are maintained at the top of the file. Then a number of functions are declared which build up the model and tensorflow operations for training and prediction. Another file (helper.py) contains functions to prepare and transform data as well as generate batches for training and test images for prediction.

* Load VGG:
	* This function loads the input, keep probability and 3 intermediate layers of the pre-trained VGG16 model.
	* These layers are then passed to the layers function to build up our model architecture.
* Layers:
	* This function creates the fully convolutional model for training as per the architecture outlined in the paper referenced above.
	* First, the 3 intermediate layers are passed through a 2d convolution (layers 3 and 4 are scaled as per the notes contained in part 1 here: https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf)
	* Each of the convolutions has l2 regularisation applied to them as suggested by the project walk through
	* Then, the 3 layers go through a pattern of transpose 2d convolutions (upsampling) and addition to one of the other layers (skip layers)
	* Each of the transpose layers also has l2 regularisation applied
	* The resulting model is returned from the function

![Architecture](/images/writeup/architecture.png)
taken from: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

* Optimize:
	* The optimise function takes the output of the last layer, the correct label tensor, the learning rate and the number of classes
	* the output layer and correct label tensors are trasnformed to 2D (although this should not be strictly necessary according to notes in https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf)
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

![No regularisation](/images/writeup/bad_reg.png)

#### Initial results

This base model (with regularisation) was then tested using various combinations of the hyper-parameters. After a limited amount of tuning, the best model loss achieved was 0.128 with 50 epochs, drop out of 0.5, a batch size of 8 and a learning rate of 5E-5. It should also be noted that to run a batch size of more than 4 required a GPU of 11GB (at least a GPU instance > 3GB)

Example output images can be seen below, showing reasonable if not perfect performance:

![overlay testing](images/writeup/test1.png)

#### Updating helper.py to pre-process the Kitti Road Dataset Images

The first optional task was to use data augmentation techniques to improve network performance. This could have taken the form of adding transformed training images to broaden the training set, which was quite small.

I referred to https://arxiv.org/pdf/1704.06857.pdf which contained a brief overview of augmentation and pre-processing steps for fully convolutional networks.

From this work i chose to trial adding images:
* had a gamma correction of 0.5
* were rotated by 180 degrees
* were flipped

This added 3 x the number of original training images to the dataset.

I also experimented with applying a gaussian function over all the base images.

The use of USE_GAUSSIAN and USE_AUGMENTATION flags in helper.py control whether either or both settings are used. The increase in data size meant that with the 11GB GPU i was using, i could only operate with a batch size of 4 and training time increased considerably.

Using this approach, training loss was reduced to 0.106. Further work would be needed here to improve performance by adding transformed images to the data set using clipping, rotation, random image enhancements such as brightness and contrast.

![overlay aug](images/writeup/test2.png)

#### Saving and Restoring the model to be run in prediction so a Video pipeline could be created

The second optional task was to create a pipeline for prediction on a video. This is implemented in prediction.py. Simply, the video is opened using moviepy and a function is passed to the movie (fl_image) which applies the pipeline to each frame before moviepy creates a new video from the altered frames output from the function.

The function opens the saved model from the training run. It captures the logits output, image tensor and keep_probability variable. Each frame is resized to the expected model image size, pre-processed inline with training (see normalisation above) and then run through the model to capture the logits output.

This output is then used as the basis of the new frame.

The output video is contained in the ./images/writeup folder

#### Adapting this approach and applying it to the cityscapes dataset

The final optional task was to adapt the model to work on the cityscapes dataset. This was not attempted due to time constraints. To complete this task i think i would have had to:

* adjust the base model to allow for the full number of classes (11?) on the dataset
* balanced the training data so that each class was represented a relatively even number of times across the input images (this would be achieved by using similar augmentation techniques as discussed above)
* a different approach to ground truth processing would have been needed as the ground truth was not binary
* a much larger processing capacity would have been required as the model parameters would have increased due to the increase in classes

#### Modes of operation

Run the following command to run the project for basic FCN on Kitti Dataset with image pre-processing:
```
python main.py
```
run ```prediction.py``` to process the video output or a single image (you will need to review the code to understand the correct paths for input / output)

### Improvements

Augmentation: Better experiments and more randomised augmentation (as per the https://arxiv.org/pdf/1704.06857.pdf paper) should lead to better generalisation.

Using a validation set could improve network performance on unseen images.

More time tuning Hyper-Parameters could lead to better performance as could implementing mean_iou to gauge the success of a particular model.

Optimisation for inference was not used in this attempt but the techniques taught in the classroom would be useful in reducing the size of the model used for inference.

### Notes from the original Udacity README

- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow.
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy.
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

