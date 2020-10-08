# Traffic Sign Classification 
---

### **Overview**
---
As we move through the skills necessary for an autonomous car to safely navigate the streets, we will land on traffic sign classification. A large part of driving safely is obeying traffic laws and using the visual queues provided by signs to anticipate upcoming road environments. An autonomous car must be able to interpret traffic signs to understand if there are new hazards or environments on or around the road ahead. Hazards may include construction zones or school zones, while new environments would be a changing speed limit or tightly curving roads.

The steps taken to build the classifier are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/sample_images.png "Sample Images"
[image2]: ./examples/class_counts.png "Class Counts Histogram"
[image3]: ./examples/augmented_images.png "Augmented Images"
[image4]: ./examples/augmented_class_counts.png "Augmented Training Data Class Counts Histogram"
[image5]: ./examples/accuracy_plot.png "Training Accuracy Plot"
[image6]: ./examples/new_test_images.png "New Test Images"
[image7]: ./examples/softmax_probabilities.png "Softmax Probabilities Histogram"


### **Methodology**
---
#### 1. Dataset Exploration

The image dataset used for training a road sign classifier is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). This dataset is made of 34,799 training samples and 12,630 testing samples of 43 different traffic sign classes. The images are color images with a 32 px wide by 32 px tall shape. Below is an array of sample images from the training set for us to see the different traffic signs in the data.

![alt text][image1]

Looking further into the dataset, we can observe the balance of classes by plotting the sample count of each class, as seen below. Its clear that the dataset contains significantly more samples of classes 1 through 13 (excluding class 6) than most other classes. We can conclude that the dataset is not well balanced, which may lead to training issues. A deep learning model trained on an imbalanced dataset may develop a poor ability to correctly classify certain samples if these samples are relatively sparse in the training data.

![alt text][image2]

I've chosen to generate new training samples for the sparse classes by employing image augmentation. I'm considering any class with fewer than 500 samples to be sparse. I have chosen to use three image augmentation techniques to alter existing samples thereby creating new images for training. These augmentation techniques are zoom, zooming in on an image, rotation, rotating the image clockwise or counter clockwise, and translation, shifting the image left, right, up, or down. These three augmentation are applied to the sparse class images in a sequential manner and applied at a random magnitude.

To illustrate the sequential manner of image augmentation: `image --> zoom() --> rotate() --> translate() --> augmented image`

Below are several examples of augmented images could exist in the new, enlarged training dataset. 

![alt text][image3]

Now that the minimum number of images in each class is 500, and the total number of training samples is 45,299. The only preprocessing step applied to all images is normalization. I've chosen to keep the color channels of the images because they contain additional information that the convnet can learn to help improve classification accuracy. To normalize images with the (R, G, B) color channels I found the mean and standard deviation of of each channel across the dataset, then normalized each channel independently using the following calculation: `norm = (x - mean)/std_dev`, where `x` is a pixel value in one of the three color channels and `mean` and `std_dev` are the mean and standard deviation of that color channel. The final distribution of classes is shown below.

![alt text][image4]

#### 2. ConvNet Model

The convnet model used in this project largely resembles the architecture of LeNet, but the convolutional layers have been made deeper on account of the use of color images rather than single channel grayscale images. I've chosen to make the convolutional layers deeper because the model will accept an input with more channels and with more layers the model has more opportunity to learn features of the images. Another modification made to the original LeNet was the addition of dropout between the first and second fully connected layers and the second and third fully connected layers. The keep rate of the the two dropout functions was 60% and 75% respectively. The addition of dropout helps improve the models robustness by preventing the model from relying too heavily on any one node for correct predictions during training.

The architecture of my LeNet2 model is as follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	    	| 2x2 stride,  outputs 5x5x32   				|
| Fully connected		| 220 nodes 									|
| RELU					|												|
| Dropout       		| 60% keep rate 								|
| Fully connected		| 84 nodes  									|
| RELU					|												|
| Dropout       		| 75% keep rate 								|
| Fully connected		| 43 nodes  									|
| Softmax				|												|

#### 3. Model Training

Model training was performed using the cross entropy loss function and the Adam optimizer. The model was trained over a course of 50 epochs with a batch size of 128 images. The learning rate of the Adam optimizer was set to 0.001. Logging the training set accuracy and validation set accuracy during during training produces the following accuracy plot. We can see that the model quickly reaches training and validation accuracies greater than 0.90. After 40 epochs the model achieves a training accuracy of 0.996 and a validation accuracy of 0.948.

![alt text][image5]

The resulting model from this training routine achieved a test set score of 0.929. During the training phase of this project, model architecture was the focus of my experimentation. Some experiments included adding another convolutional layer between the first layer and the following pooling layer to further increase the number of feature maps in the model. I also experimented with increasing the number of nodes in the fully connected layers. I found that neither of these changes significantly improved my model over my current LeNet2 design.

#### 4. Test on New Images

The final step in this project was to test the model on hand selected traffic sign images from the web. The only criteria used when searching to test images were, images of sign classes that exist in the training data and images that are relatively square. The test images would have to be resized to 32x32, so rectangular images would be skewed. Below are 7 images found from a google search. I suspect these images will be mostly easy for the model to correctly classify because the sign is centered in most images and takes up most of the frame.

![alt text][image6]

The overall accuracy of the model on the new test images was lower than expected with an accuracy of 0.86. In other words, 6 of 7 classes were predicted correctly. The model predictions are listed in the table below:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| **50 KPH Speed Limit**	| **30 KPH Speed Limit**							| 
| Priority Road 		| Priority Road 								|
| Roundabout			| Yield											|
| Bumpy Road     		| Bumpy Road					 				|
| Road Work Ahead		| Road Work Ahead      							|
| No Entry      		| No Entry          							|
| Stop Sign     		| Stop Sign         							|

We can look closer into these predictions by examining the top 5 softmax probabilities for each image. The below graphic shows each input image and an adjacent histogram showing the softmax probabilities. We can see that the model predicts the class for 6 of the 7 images with 100% probability, even for the incorrectly classified image. The model was the least certian for the No Entry sign image where it returned an 80% probability for the No Entry class and a 20% probability for the Priority Road class. I'm surprised that the incorrectly classified speed limit sign didn't produce a distribution of probabilities for the first few speed limit sign classes. This leads me to think that the model hasn't thoroughly learned some of the distinguishing features for different speed limit signs.

![alt text][image7]
