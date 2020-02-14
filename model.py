import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import sklearn 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from math import ceil
import time

#Extracting the data from the spreadsheet
lines = []
#print("Loading data...")
#with open('data/driving_log.csv') as csvfile:
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None) #this is necessary to skip the first record as it contains the headings

    for line in reader:
        lines.append(line)
#print("Data saved")     

#Splitting daata in the training and validation sets
#80 vs 20 split
train_samples, validation_samples = train_test_split(lines , test_size=0.2)
# print("Number of training samples: ", len(train_samples))
# print("Number of validation samples: ", len(validation_samples))

startTime = time.time()


def generator(samples, batch_size=32):
    it = 1 #tracking iterations 
    num_samples = len(samples)
    print("Number of samples: ", num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
        
            images =  []
            steering_angles = []
            
            for batch_sample in batch_samples:
                #for i in range(3) #collating 3 camera angle images               
               
                
                # 3 camera angles 
                cameras = ['center', 'left', 'right']
                for camera in cameras:                    
                    
                    steering_angle = float(batch_sample[3])
                    #print(len(batch_sample[0]))
                    # adjust the steering angle for the left and right side cameras
                    #aids in recovery 
                    if camera == 'center':
                        #print(batch_sample[0])
                        
                        path = '/opt/carnd_p3/data/IMG/'+batch_sample[0].split('/')[-1]
                        #path = 'data/IMG/'+batch_sample[0].split('/')[-1]
                        #steering_angle = float(batch_sample[3])
                        steering_angle += 0        
                    elif camera == 'left':
                        #print(batch_sample[1])
                        path = '/opt/carnd_p3/data/IMG/'+batch_sample[1].split('/')[-1]

                        #path = 'data/IMG/'+batch_sample[1].split('/')[-1]
                        #steering_angle = float(batch_sample[3])
                        steering_angle += 0.2
                    elif camera == 'right':
                        #print(batch_sample[2])
                        path = '/opt/carnd_p3/data/IMG/'+batch_sample[2].split('/')[-1]
                            
                        #path = 'data/IMG/'+batch_sample[2].split('/')[-1]
                        #steering_angle = float(batch_sample[3])
                        steering_angle -= 0.2
                    
                    #source_path = batch_sample[0]
                    #img_name = source_path.split('/')[-1]
                    #current_path = 'data/IMG/' + img_name
                    #print(path)
                    image = cv2.imread(path)
                    #convert from BGR to RGB
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
                    #Error checking 
                    #Udacity data has more image files than rows in the spreadsheet provided
                    if image is None:
                        print("WARNING Images %s at row number %d is None"%(path,it))
                        it += 1
                        continue
    
                    images.append(image)
                    steering_angles.append(steering_angle)
                    it +=1
                    
                #Flipping the images horizonally to counteract left bias
                #Doubling the data
                augmented_images, augmented_angles = [],[]

                for image, steering_angle in zip(images,steering_angles):
                    augmented_images.append(image)
                    augmented_angles.append(steering_angle)
                    augmented_images.append(cv2.flip(image,1)) #positive value (for example, 1) means flipping around y-axis
                    augmented_angles.append(steering_angle*-1.0)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)     
            
            yield sklearn.utils.shuffle(X_train, y_train)

print("Training the model ...")
        
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)            
     
#Model
#Based on the NVIDIA CNN   
#See link    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
#Lambda function normalises the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320,3)))
model.add(Cropping2D(cropping=((65,25), (0,0)))) #50 rows from top, 20 rows from bottom
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.50)) #Dropout layer 50% reduces overfitting
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1)) #width 1 

model.compile(loss="mse", optimizer="adam")
#model.fit(X_train, y_train, validation_split=0.2,  shuffle=True, epochs=7)
History_Model = model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

### print the keys contained in the history object
#print(History_Model.history.keys())

### plot the training and validation loss for each epoch
plt.plot(History_Model.history['loss'])
plt.plot(History_Model.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

print("Read date in %0.2f seconds"%(time.time() - startTime),flush=True)

model.save('model_2.h5')
print('model saved as model_2.h5')