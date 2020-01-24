# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:14:28 2020

@author: LOVESA
The script used to create and train the model.


"""
#conda install -c conda-forge keras

import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sklearn 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from math import ceil

samples = []
csv_path = "data/driving_log_SL.csv"
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    #next(reader, None) #this is necessary to skip the first record as it contains the headings

    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples , test_size=0.2)

#startTime = time.time()

def generator(samples, batch_size=32):
    it = 1
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                
                if center_image is None:
                    print("WARNING Images %s at row number %d is None"%(name,it))
                    it += 1
                    continue
                
                center_angle = float(batch_sample[3])
                images.append(center_image)
                steering_angles.append(center_angle)
                it += 1
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            #plt.imshow(X_train[0].shape)

            yield sklearn.utils.shuffle(X_train, y_train)

#print("Read date in %0.2f seconds"%(time.time() - startTime),flush=True)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#ch, row, col = 3, 80, 320 # Trimmed image format

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#print(X_train.shape,flush=True)

model = Sequential() 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320,3))) 
#model.add(Lambda(lambda x: (x / 127.5) - 0.5, input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Cropping2D(cropping=((50,20), (0,0)))) #50 rows from top, 20 rows from bottom

model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.50)) #Dropout layer 50% reduces overfitting
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1)) #width 1 

model.compile(loss="mse", optimizer="adam") 
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True,epochs=7, verbose=1)
#minimise error between steering prediction and actual
History_Model = model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

### print the keys contained in the history object
print(History_Model.history.keys())

### plot the training and validation loss for each epoch
plt.plot(History_Model.history['loss'])
plt.plot(History_Model.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#default is 10epochs
model.save('model.h5')

