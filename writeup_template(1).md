# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./train.jpg "Visualization"
[image2]: ./test-data/end.jpg "Traffic Sign 1"
[image3]: ./test-data/right.jpg "Traffic Sign 2"
[image4]: ./test-data/traffic.png "Traffic Sign 3"
[image5]: ./test-data/work.png "Traffic Sign 4"
[image6]: ./test-data/yield.png "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my project code:(https://github.com/AbishekNP/TrafficSignClassifier)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is :34799
* The size of test set is : 12630
* The shape of a traffic sign image is : 932,32,3)
* The number of unique classes/labels in the data set is : 43



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data because it can avoid the influence of  high frequency noise and very low noise. And on the other hand ,it make image data satisfy  nomal distribution. In general，ormalized image mean =0 and variance =1

 
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image, output 28x28x6				| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  same padding, outputs 14x14x6 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  same padding, outputs 5x5x16 	|
| FLATTEN				|												|
| Fully connected		| input 400, output 120							|
| Fully connected		| input 120, output 84							|
| Fully connected		| input 84, output 43							|
| Softmax				| 												|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
Epochs : 50
Learing Rate :  0.01
Batch Size : 128


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of : 0.956
* test set accuracy of : 0.8


If a well known architecture was chosen:
* What architecture was chosen : I've chose the LeNet for this task.
* Why did you believe it would be relevant to the traffic sign application : It's architecture is quite easy to understand and      also as it's a tyoically small architecture with lesser number of    layers, It makes it easier to train it using a normal GPU. This net also has a good history on classifyning images as it gave good results on the MNIST dataset.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

### Further improvements to the model:

1) Grayscaling/Segmenting the image might help.
2) If GPU's were not a problem, then we could opt for better and bigger network like Inception, Resnet-50 etc....
3) Transfer Learning could be performed.
4) Cropped images could have been used during the training phase.