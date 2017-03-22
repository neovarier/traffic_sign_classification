#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train_hist.png "Training Histogram"
[image2]: ./examples/valid_hist.png "Validation Histogram"
[image3]: ./examples/train_aug_hist.png "Augmented Training Histogram"
[image4]: ./examples/valid_aug_hist.png "Augmented Valid Histogram"
[image5]: ./examples/color.png "Original Image"
[image6]: ./examples/gray.png "Grayscale Image"
[image7]: ./examples/vanalla_loss.png "Loss"
[image8]: ./examples/vanilla_accu.png "Acuuracy"
[image9]: ./examples/reg_loss.png "Loss with L2 Regularization"
[image10]: ./examples/reg_accu.png "Accuracy with L2 Regularization"
[image11]: ./examples/loss.png "Final Loss"
[image12]: ./examples/accu.png "Final Accuracy"
[image13]: ./web_images/image1.png "Traffic Signal 1"
[image14]: ./web_images/image2.jpg "Traffic Signal 2"
[image15]: ./web_images/image3.jpg "Traffic Signal 3"
[image16]: ./web_images/image4.jpg "Traffic Signal 4"
[image17]: ./web_images/image5.png "Traffic Signal 5"
[image18]: ./web_images/image6.jpg "Traffic Signal 6"
[image19]: ./web_images/image7.png "Traffic Signal 7"
[image20]: ./web_images/image8.png "Traffic Signal 8"
[image21]: ./web_images/image9.png "Traffic Signal 9"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README
Here is the link for my implemented code:
(https://github.com/neovarier/traffic_sign_classification/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

The code for this step is contained in the second code cell of the IPython notebook.  

Number of training examples = 34799.
Number of validation examples = 4410.
Number of testing examples = 12630.
Image data shape = (32, 32, 3).
Number of classes = 43.

####2. Exploratory Visualization of the dataset

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]

###Design and Test a Model Architecture

####Data augmentation

The funtions for this step is contained in the fifth code cell of the IPython notebook.
The function execution is carried out in 6th and 7th cells for training and validation respectively.
As it can be seen in the previous plots, the histogram is not equalized and is skewed.
In training data set, the class 2 images are more than 2000 whereas the class 0 images are less than 250.
The data set should have even distribution for all the classes.
Otherwise the network will learn the features of only those classes with higher images.
It may not learn the features of the classes with less images. Thus will not be able to classify the images of classes
with lesser training data.

To equalize the histogram, the number of images of classes with lesser data are increased.
I have increased their number by applying rotation and translation.
I have tried to increase image count of the classes with less image to come close to the class with highest images
This brings variability in the dataset for those classes.
These operations are chosen as to simulate real world scenario as the images could be captured
from various locations and orientations on the road.
The maximum rotation is restricted between -3 to +3 degrees.
The maximum translation is restricted between -3 to 3 pixels.


The augmented data is saved for later use.
After data augmentation the dataset is analysed and visualised again to see the difference from the previous data set

Analysis: (Cell 11)
Number of training examples = 74217.
Number of validation examples = 9390.
Number of testing examples = 12630.
Image data shape = (32, 32, 3).
Number of classes = 43.

Visualization: (Cell 12)

![alt text][image3]
![alt text][image4]

The histogram has certainly improved

###Preprocessing
As a first step, I decided to convert the images to grayscale because the images are not dependent on color.
There are cases in real world where the same signs are depicted with different color, though they mean the same.
Also, the reduces the depth of the input data, which eventually will reduce the filter depth in the first convolution layer.
Hence would reduce the computations
An example grayscale operation

![alt text][image5]
![alt text][image6]

Next, I normalized the image data using cv2.normalize, because the pixel values range from 0 to 255.
With different images the max and min pixel values could be different. And the mean is non-zero.
Normalization would make it centered around zero and scale the range pixel within -1 to 1.
This would help the network to reach the minima faster as the feature axes in the feature hyperspace have a uniform scale accross.

The preprocessing step is carried out in cell 14.

####3. Model architecture

The code for my initial model is located in the 16th cell of the ipython notebook. 

I chose the LeNet as my initial model as it is a well proven architicture for various image based classification problems

| Layer         		      |     Description	        					                            | 
|:---------------------:|:--------------------------------------------------------:| 
| Input         		      | 32x32x1 RGB image   							                              |  
| Convolution 5x5     	 | 1x1 stride, VALID padding, outputs 28x28x6 	             |
| RELU					 												|                                                          |
| Max pooling	2x2      	| 2x2 stride, VALID padding  outputs 14x14x6 				          |
| Convolution 5x5	      | 1x1 stride, VALID padding  outputs 10x10x16      								|
| RELU					 												|                                                          |
| Max pooling	2x2      	| 2x2 stride, VALID padding  outputs 5x5x6 				            |
| Fully connected		400  | output 120        									                              |
| RELU					 												|                                                          |
| Fully connected		120  | output 84        									                               |
| RELU					 												|                                                          |
| Fully connected		84   | output 10        									                               |
| Softmax				           |             									                                    |

 


####4. Hyperparameter and optimizer 

Optimizer: Adam. 
I used Adam as it automatically takes care of decreasing the learning rate
so that it can the coarser steps in the beginning and take finer steps in the end to land at the global minima accurately.
Batch size: 128
I used 128, as this the maximum that my GPU was allowing. This would achieve maximum parallization in the computation.
Learning Rate: 0.001
I chose a small learning rate so that it does not skip the global minima.
Epochs: 150
I started with high number of epochs to analyse the pattern of training and validation loss.



The code for training the model is located in the 20th cell of the ipython notebook. 


####5. Solution

I tracked the following while training:
* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

I tracking the training and validation loss to see if the model was overfitting.
I tracked the training and validation accuracy to see if the accuracy increases and reaches the expected value.

With the above tracking I saw that the the model was overfitting.

![alt text][image7]
![alt text][image8]

The validation loss reduces first and then it keeps increasing, whereas the training loss keeps decreasing.

To prevent overfitting I emplyed the following techniques:
* L2 loss regularization
I tried L2 loss regularization beta with 0.1,0.01 & 0.001.
With 0.001, the loss was reducing but not smoothly. The accuracy was also not going beyond 0.9.

![alt text][image9]
![alt text][image10]

* Dropouts
In conjunction with L2 loss, I tried applying dropouts in just the fully connected layer with keep_prob=0.5, but was not giving expected accuracy. Then tried dropouts for all layers with (including convolutional layers). With this I noticed that I had to increase
the keep_prob for increasing the accuracy. But still was not getting greater than 0.93 validation accuracy. I tried to take the best keep_prob that works for each conv and fc layers separately. 
This gave me better results:
conv keep_prob:0.9
fc keep_prob: 0.5
It seemed that high dropouts on conv layers does not help. With this configuration the validation accuracy was exceeding 0.93.
* Early stopping
I employed early stopping as it was achieving desired accuracy much before 150 epoch.

![alt text][image11]
![alt text][image12]

The training and validation accuracy were calulated in cell 20.
The testing accuracy is calculated in cell 24.
The logs for the network training is captured in:
https://github.com/neovarier/traffic_sign_classification/blob/master/log.txt

My final model results were:
* training set accuracy of = 99.2
* validation set accuracy of = 94.3
* test set accuracy of = 94.3

In validation accuracy, after approaching 94.3 it reduces further and it does not pick as much in the next 15 epochs.
To increase the accuracy further may I can apply pretrained weights at 94.3 and train again.

###Test a Model on New Images

Here are nine German traffic signs that I found on the web:

![alt text][image19] ![alt text][image16] ![alt text][image14] 
![alt text][image20] ![alt text][image18] ![alt text][image21]
![alt text][image15] ![alt text][image13] ![alt text][image17]

The image 2 is a Pedestrian image, might difficult to classify because of the presence of the zebra crossing, which is not present in the training data. Also the traffic signal shape for image 9 is different from the training set as it is not triangular.
The code for making predictions on my final model is located in the 31st cell of the Ipython notebook.

Here are the results of the prediction:

| Image			              |     Prediction	        					| 
|:---------------------:|:---------------------------:| 
| Bicycles crossing     | Bicycles crossing   								| 
| Bumpy road     			    | Bumpy road 										       |
| Stop					             | Stop											             |
| Keep left	      		    | Keep left					 				         |
| General caution			    | General caution      							|
| Traffic signals       | Traffic signals             |
| Pedestrians           | Pedestrians                 |
| Children crossing     | Children crossing           |
| Wild animals crossing | Dangerous curve to the left |


The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88.8889%. This compares favorably to the accuracy on the test set of test data which is 94.3.
The image 9 which is "Wild animals crossing" whose shape is similar to "Dangerous curve to the left".
In the top 5 softmax probablities, the correct prediction does come up but at the 3rd rank.

The code for outputting the softmax probabilities for the 9 traffic signals is located in the 32th cell of the Ipython notebook.

Image  1

| Probablity		          |     Prediction	        					| 
|:---------------------:|:---------------------------:| 
| 0.979586              | Bicycles crossing           |
| 0.0196223             | Bumpy road                  |
| 0.000722326           | Children crossing           |
| 4.91705e-05           | Slippery road               |
| 1.46039e-05           |Beware of ice/snow           |

Image  2

| Probablity		          |     Prediction	        					| 
|:---------------------:|:---------------------------:| 
| 0.999995              | Bumpy road                  |
| 5.45566e-06           | Bicycles crossing           |
| 1.93473e-09           | Traffic signals             |
| 4.78361e-11           | Road work                   |
| 2.07294e-11           | Beware of ice/snow          |

Image  3

| Probablity		          |     Prediction	        					| 
|:---------------------:|:---------------------------:| 
| 0.999866              | Stop                        |
| 9.31516e-05           | Yield                       |
| 3.94518e-05           | Keep right                  |
| 1.59648e-06           | Speed limit (50km/h)        |
| 7.91644e-08           | Turn left ahead             |

Image  4

| Probablity		          |     Prediction	        					        | 
|:---------------------:|:-----------------------------------:| 
| 1.0                   | Keep left                           |
| 2.45996e-19           | End of no passing                   |
| 1.59154e-19           | Road work                           |
| 3.30148e-20           | End of all speed and passing limits |
| 1.44789e-20           | Turn right ahead                    |

Image  5

| Probablity		          |     Prediction	        					        | 
|:---------------------:|:-----------------------------------:| 
| 0.993743              | General caution                     |
| 0.00609989            | Traffic signals                     |
| 0.000155608           | Pedestrians                         |
| 6.43928e-07           | Road work                           |
| 4.06901e-07           | Bumpy road                          |

Image  6

| Probablity		          |     Prediction	        					        | 
|:---------------------:|:-----------------------------------:| 
| 0.987753              | Traffic signals                     |
| 0.012234              | General caution                     |
| 1.22218e-05           | Pedestrians                         |
| 2.00876e-07           | Bumpy road                          |
| 3.10773e-08           | Road work                           |

Image  7

| Probablity		          |     Prediction	        					        | 
|:---------------------:|:-----------------------------------:| 
| 0.688249              | Pedestrians                         |
| 0.240953              | Children crossing                   |
| 0.0628403             | Road narrows on the right           |
| 0.0031001             | Bicycles crossing                   |
| 0.00255504            | Beware of ice/snow                  |

Image  8

| Probablity		          |     Prediction	        					         | 
|:---------------------:|:------------------------------------:| 
| 0.999179              | Children crossing                    |
| 0.000476159           | Bicycles crossing                    |
| 0.000311066           | Beware of ice/snow                   |
| 1.74707e-05           | Dangerous curve to the right         |
| 6.38499e-06           | Right-of-way at the next intersection|

Image  9

| Probablity		          |     Prediction	        					         | 
|:---------------------:|:------------------------------------:| 
| 0.99846               | Dangerous curve to the left          |
| 0.00074877            | Go straight or left                  |
| 0.00029424            | Wild animals crossing                |
| 0.000198753           | Slippery road                        |
| 0.000198099           | Speed limit (70km/h)                 |

