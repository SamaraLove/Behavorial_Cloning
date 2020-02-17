**Behavioral Cloning Project**
Self Driving Car Udacity Nanodegree

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Flipped_img.png "Flipped Image"
[image2]: ./examples/Cropped.png "Cropped image"

The image example used above is center_2016_12_01_13_30_48_287.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_2.h5 the saved model, containing a trained convolution neural network 
* writeup_report_SLP4.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python model2.py
python drive.py model_2.h5

**Saving the video**
python drive.py model_2.h5 run1
python video.py run1

```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes (model.py). 


#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer of 50% in order to reduce overfitting (model.py). 

The model was trained and validated on the full udacity provided dataset. It was also verified on a small dataset that I obtained. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py).


#### 4. Appropriate training data

I used the dataset provided by Udacity which had 24108 images. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one provided in the tutorials, from there I based the model on the NVIDIA CNN, see the link below   http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

Each layer was changed one at a time through a trial and error process to determine how best the car drove autonomously.

The data is generated for training using a 'generator' to speed up the process by not storing the training data in memory. 

The data is normalized in the model using a Keras lambda layer.

The model includes 4 x 'relu' layers to introduce nonlinearity. 

The final model acheived a low mean squared error on all training and valdiation sets which proved accurate when driving around the track autonomously without leaving the course.


#### 2. Final Model Architecture

The final model architecture (in model.py) consisted of a convolution neural network with the following layers, as seen in the model.py code snippet below:

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320,3))) #Normalising data
model.add(Cropping2D(cropping=((65,25), (0,0)))) #Cropping 65 rows from top, 25 rows from bottom
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dense(100, activation='relu')) #width 100
model.add(Dropout(0.50)) #Dropout layer 50% reduces overfitting
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1)) #width 1


#### 3. Creation of the Training Set & Training Process
I used the Lambda keras layer to normalize.

*Horizontally Flipping the images*
The data was baised towards turning left, so the images were horizonally flipped and the steering angles were flipped opposite too. This also doubled the dataset. 
![alt text][image1]

*Cropping the images* 
The images were cropped from 65 rows from top and 25 rows from bottom. This was done to minimise the distractions and alternate objects that could trigger the model to drive in an undesired way. 
![alt text][image2]

*Left and right cameras*
Initially, just the centre camera was used to train the model, but on the sharper corners, it was noticeable that the car was too slow to react and turn whilst staying in the centre of the road. The data was biased in one direction, so we utilised the left and right cameras with an offset tuned to 0.2. 0.2 was chosen as any higher, and the vehicle would jerk too much around the corners and would oversteer. 

After the collection process, I had 24108 x 2 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used an adam optimizer so that manually training the learning rate wasn't necessary. 5 epochs was chosen to reduce the time required for trialling the model and enabled faster trial and error discoveries with tuning the model. 
