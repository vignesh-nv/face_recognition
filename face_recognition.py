import requests
import random
import numpy as np 
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt
import urllib
import cv2
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def saveMaleImage(image):
  global malecount
  image = image.convert('RGB')
  if malecount<70:
    name = 'Images/Train/Male/'+str(malecount)+'.jpeg' 
    image.save(name)
  else:
    name = 'Images/Validation/Male/'+str(malecount)+'.jpeg' 
    image.save(name)
  malecount+=1
 
def saveFemaleImage(image):
  global femalecount
  image = image.convert('RGB')
  if femalecount<80:
    name = 'Images/Train/Female/'+str(femalecount)+'.jpeg'
    image.save(name)
  else:
    name = 'Images/Validation/Female/'+str(femalecount)+'.jpeg' 
    image.save(name)
  femalecount+=1
  
#Downloading and preprocessing the Images 
def downloadTraining(df):
    for index, row in df.iterrows():
        # Get the image from the URL
        resp = urllib.request.urlopen(row[1])
        im = np.array(Image.open(resp))
        
        for l in range(len(row[0])):
          p = row[0][l]['points']
          annotations = row[0][l]['label']
          print(annotations)
          # Points of rectangle
          x_point_top = p[0]['x']*im.shape[1]
          y_point_top = p[0]['y']*im.shape[0]
          x_point_bot = p[1]['x']*im.shape[1]
          y_point_bot = p[1]['y']*im.shape[0]
          carImage = Image.fromarray(im)
          print(carImage)
          croppedImage = carImage.crop((x_point_top, y_point_top, x_point_bot, y_point_bot))
          
          if 'G_Male' in annotations:
            saveMaleImage(croppedImage)
          elif 'G_ Female' in annotations:
            saveFemaleImage(croppedImage)

#Loading Json File
data = pd.read_json('Face_Recognition.json')
pd.set_option('display.max_colwidth', -1)
data['points'] = data.apply(lambda row: row['annotation'][0]['points'], axis=1)

malecount = 0
femalecount = 0
downloadTraining(data)

#Declaring the Image paths and size for ML model
BATCH_SIZE = 5
image_size = 160
train_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)
base_dir = 'Images'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')

#Training Data
train_data = train_image_generator.flow_from_directory(batch_size = BATCH_SIZE,
                                                     directory = train_dir, 
                                                     shuffle = True,
                                                     target_size = (image_size, image_size),
                                                     class_mode = 'binary')
#Validation Data
validation_data = test_image_generator.flow_from_directory(batch_size = BATCH_SIZE, 
                                                          directory = validation_dir,
                                                          shuffle = False, 
                                                          target_size = (image_size, image_size),
                                                          class_mode = 'binary')

#Machine Learning Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

EPOCHS = 20
history = model.fit_generator(
    train_data,
    steps_per_epoch=int(np.ceil(150 / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=validation_data,
    validation_steps=int(np.ceil(21 / float(BATCH_SIZE)))
)
