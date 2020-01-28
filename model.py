# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:51:34 2020

@author: LOVESA
"""

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images =  []
steering_angles = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    print(current_path)
    
    image = cv2.imread(current_path)
    images.append(image)
    
    steering_angle = float(line[3])
    steering_angles.append(steering_angle)

augmented_images, augmented_angles = [],[]

for image, measurement in zip(images,steering_angles):
    augmented_images.append(image)
    augmented_angles.append(steering_angles)
    augmented_images.append(cv2.flip(image,1)) #positive value (for example, 1) means flipping around y-axis
    augmented_angles.append(steering_angle*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_angles)     
    
#X_train = np.array(images)
#y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D



model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)))) #50 rows from top, 20 rows from bottom
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.50)) #Dropout layer 50% reduces overfitting
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1)) #width 1 

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=0.2,  shuffle=True, nb_epoch=7)

model.save('model.h5')
