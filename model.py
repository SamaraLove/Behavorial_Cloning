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

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader, None) #this is necessary to skip the first record as it contains the headings

    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples , test_size=0.2)

def generator(samples, batch_size=32):
    it = 1 #tracking iterations 
    num_samples = len(samples)
    print(num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
        
            images =  []
            steering_angles = []
            
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                print(name)

                if center_image is None:
                    print("WARNING Images %s at row number %d is None"%(name,it))
                    it += 1
                    continue

                steering_angle = float(batch_sample[3])
                images.append(center_image)
                steering_angles.append(steering_angle)
                                
                augmented_images, augmented_angles = [],[]

                for image, measurement in zip(images,steering_angles):
                    augmented_images.append(center_image)
                    augmented_angles.append(steering_angle)
                    augmented_images.append(cv2.flip(image,1)) #positive value (for example, 1) means flipping around y-axis
                    augmented_angles.append(steering_angle*-1.0)
                it +=1 
            
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)     

            #X_train = np.array(images)
            #y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)            
            
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)))) #50 rows from top, 20 rows from bottom
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
#model.add(Dense(100, activation='relu'))
model.add(Dropout(0.50)) #Dropout layer 50% reduces overfitting
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1)) #width 1 

model.compile(loss="mse", optimizer="adam")
#model.fit(X_train, y_train, validation_split=0.2,  shuffle=True, epochs=7)

History_Model = model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

### print the keys contained in the history object
print(History_Model.history.keys())

model.save('model.h5')
print('model saved')
