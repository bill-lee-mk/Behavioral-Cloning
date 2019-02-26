# -*- coding: utf-8 -*-
"""
Created on Sun May  6 17:10:18 2018

@author: Bill-LI
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import cv2
#import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, Cropping2D
from keras.layers.core import Lambda
from keras.optimizers import Adam

# Pre-process images before feed to train, including crop, resize, YUV, grayscale
# Pre-process steps are taken inside the model to better apply to new input images
# This project doesn't use process_image function but list below for further reference
'''
def process_image(image): 
    
    #image = Image.open('data/IMG/center_2016_12_01_13_30_48_287.jpg')
    #plt.imshow(image)
    #print (image.format, image.size, image.mode)
    
    #image_cropped = image.crop((0,70,320,140))
    #plt.imshow(image_cropped)
    #print (image_cropped.format, image_cropped.size, image_cropped.mode)
    
    #image_resized = image_cropped.resize((200,66))
    
    #plt.imshow(image_resized)
    #print (image_resized.format, image_resized.size, image_resized.mode)
    
    # change colour/brightness to better generalize model
    #image_out = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)
    image_out = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    #plt.imshow(image_out)
    #print (image_out.shape)    
    
    return image_out
'''
    

def get_data(csv_data_file):
    lines = []
    with open (csv_data_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines[1:] # exclude headers
                    
    

def train_data(csv_data_file):
    
    images_center, images_left, images_right = [], [], []
    steers_center, steers_left, steers_right = [], [], []
    correction_factor = 0.2
    
    for line in get_data(csv_data_file):

        # expand steer angles for side camera images by +/- bias
        steer_center = float(line[3])
        steers_center.append(steer_center)
        
        steer_left = float(line[3])+correction_factor
        steers_left.append(steer_left)
        
        steer_right = float(line[3])-correction_factor
        steers_right.append(steer_right)        
          
        # Use side camera images to expand training set by 3 times, 
        # Benefits: 1) more data to train 
        #           2) teach network to steer back to center if drift off the side
          
        source_path = line[0]
        filename = source_path.split('/')[-1]
        file_path = 'data/IMG/' + filename
        images_center.append(file_path)
    
        source_path = line[1]
        filename = source_path.split('/')[-1]
        file_path = 'data/IMG/' + filename
        images_left.append(file_path)
                    
        source_path = line[2]
        filename = source_path.split('/')[-1]
        file_path = 'data/IMG/' + filename
        images_right.append(file_path)
                   
        total_steers = steers_center + steers_left + steers_right
        total_images = images_center + images_left + images_right
      
    return (total_images, total_steers)
            

# Indefinitely yield batches of training data
def batch_generator(samples, batch_size):
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images, angles = [], []
            
            for imagePath, measurement in batch_samples:
                # cv2 load image in BGR colorspace
                originalImage = cv2.imread(imagePath) 
                # drive.py load images in RGB to predict the steering angles.
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB) 
                images.append(image)
                angles.append(measurement)
                
                # Data augumentation to expand training set by 2 times
                # teach the car to steer clockwise and counter-clockwise
                # flip images horizontally and invert steer angles 
                # Benifis: 1) more data to train 2) data is more comprehensive
                
                #if abs(float(measurement))>=0.05:
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)            

def resize(image):
    import tensorflow as tf  
    return tf.image.resize_images(image, (66, 200))

def PreProcessingLayers():
    
    model = Sequential()
    # Crop 70 pixels from the image top and 25 from the bottom
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    # resize mage to (66,200) for Nvidia model
    model.add(Lambda(resize))
    # Normalizae to [-0.5, +0.5]
    model.add(Lambda(lambda x: (x/255.0) - 0.5))
    return model
    
def Nvidia_Model():

    model = PreProcessingLayers()
    model.add(Conv2D(24,(5,5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), padding='valid', activation='relu'))
    model.add(Conv2D(64,(3,3), padding='valid', activation='relu'))
    model.add(Conv2D(64,(3,3), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    #model.add(Dropout(0.2))
    model.add(Dense(50))
    #model.add(Dropout(0.2))
    model.add(Dense(10))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    model.summary()
    
    return model


total_images, total_steers = train_data('data/driving_log.csv')
print('Total Number of Images:', len(total_images))
print('Total Number of Angles:', len(total_steers))

# Split samples and create generator.
samples = list(zip(total_images, total_steers))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Train samples:', len(train_samples))
print('Validation samples:', len(validation_samples))

train_generator = batch_generator(train_samples, batch_size=32)
validation_generator = batch_generator(validation_samples, batch_size=32)


# Create model
model = Nvidia_Model()

# Compile model
model.compile(loss='mse', optimizer = Adam(lr = 0.0001), metrics=['accuracy'])

# Train model
batch_size = 512
nb_epoch = 3

history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch= math.ceil(len(train_samples)/batch_size), 
                                     epochs=nb_epoch, 
                                     verbose=1, 
                                     validation_data=validation_generator, 
                                     validation_steps=math.ceil(len(validation_samples)/batch_size), 
                                     )

# Save model
model.save('model.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:  
    json_file.write(model_json)  
print("Saved model to disk")



print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])


# plot the training and validation loss across epochs
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper right')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.show()


# plot the training and validation accuracy across epochs
plt.plot(history_object.history['acc'])
plt.plot(history_object.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='upper right')
plt.grid(color='black', linestyle='--', linewidth=1)
plt.show()


