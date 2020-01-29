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
#from math import ceil
import pandas as pd

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader, None) #this is necessary to skip the first record as it contains the headings

    #Reads the csv file 
    #csv_data = pd.read_csv(csvfile, header=None, names=['center', 'left', 'right', 'angle','throttle', 'break', 'speed'])
    #Use the csv.DictReader() object to make it easier to select a column by the header:
    #https://stackoverflow.com/questions/25341417/find-a-specific-header-in-a-csv-file-using-python-3-code
    
    for line in reader:
        samples.append(line)
        #get_col = list(pd.read_csv(csvfile,sep="|",nrows=1).columns)
        #print(get_col)
        #https://stackoverflow.com/questions/24962908/how-can-i-read-only-the-header-column-of-a-csv-file-using-python/38675183
        
train_samples, validation_samples = train_test_split(samples , test_size=0.2)

def generator(samples, batch_size=32):
    it = 1 #tracking iterations 
    num_samples = len(samples)
    print("Number of samples: ",num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
        
            images =  []
            steering_angles = []
            
            for batch_sample in batch_samples:
                #for i in range(3) #collating 3 camera angle images
                
                #READ IN THE FROM THE DIFFERENT CAMERA ANGLES
                #potential ways to do this above
                #how do I make this work?
                
                # 3 camera angles 
                camera = ['center', 'left', 'right']
                
                steering_angle = float(batch_sample[3])
                
                # adjust the steering angle for the left and right side cameras
                #aids in recovery 
                if camera == 'center'
                    path = 'data/IMG/'+batch_sample[0].split('/')[-1]
                    steering_angle += 0
                elif camera == 'left':
                    path = 'data/IMG/'+batch_sample[1].split('/')[-1]
                    steering_angle += 0.25
                elif camera == 'right':
                    path = 'data/IMG/'+batch_sample[2].split('/')[-1]
                    steering_angle -= 0.25
                
                #source_path = batch_sample[0]
                #img_name = source_path.split('/')[-1]
                #current_path = 'data/IMG/' + img_name
                #print(path)
                image = cv2.imread(path)

                #Error checking 
                #Udacity data has more image files than rows in the spreadsheet provided
                if image is None:
                    print("WARNING Images %s at row number %d is None"%(img_name,it))
                    it += 1
                    continue

                images.append(image)
                steering_angles.append(steering_angle)
                             
                #Flipping the images horizonally to counteract left bias
                #Doubling the data
                augmented_images, augmented_angles = [],[]

                for image, measurement in zip(images,steering_angles):
                    augmented_images.append(image)
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
