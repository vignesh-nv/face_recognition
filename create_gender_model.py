from six.moves import urllib
import os
import tarfile
import datetime
import cv2
import tarfile
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import shutil,math
import scipy.io

#Download the Images
URL = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar'
filePath = 'wiki.tar.gz'
urllib.request.urlretrieve(URL, filePath)

#Extract the tar file
file = tarfile.open('wiki_crop.tar')
os.mkdir('Original_Images')
file.extractall(path='Original_Images')
file.close()


#Folders for Training the Model
os.mkdir('Images')
os.mkdir('Images/Train')
os.mkdir('Images/Train/Male')
os.mkdir('Images/Train/Female')
os.mkdir('Images/Validation')
os.mkdir('Images/Validation/Male')
os.mkdir('Images/Validation/Female')


#Load the matlab file
mat = scipy.io.loadmat('Original_Images/wiki_crop/wiki.mat')

#Place the images in Training and Validation Folders
gender = []
images = []
data = mat['wiki'][0]

for v in data:
  gender = v[3][0]
  images = v[2][0]
  
#0-Female 1-Male
for i in range(60000):
  image_path = 'CroppedImages/wiki_crop/'+images[i][0]
  if score[i]==math.inf:
    continue
  if gender[i]==1:
    shutil.copy(image_path, 'Images/Train/Male')
  else:
    shutil.copy(image_path, 'Images/Train/Female')
    
for i in range(60000,62328):
  image_path = 'CroppedImages/wiki_crop/'+images[i][0]
  
  if score[i]==math.inf:
    continue
  if gender[i]==1:
    shutil.copy(image_path, 'Images/Validation/Male')
    
  else:
    shutil.copy(image_path, 'Images/Validation/Female')


BATCH_SIZE = 100
image_size = 150
train_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)
base_dir = 'Images'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')

train_data = train_image_generator.flow_from_directory(batch_size = BATCH_SIZE,
                                                     directory = train_dir, 
                                                     shuffle = True,
                                                     target_size = (image_size, image_size),
                                                     class_mode = 'binary')

validation_data = test_image_generator.flow_from_directory(batch_size = BATCH_SIZE, 
                                                          directory = validation_dir,
                                                          shuffle = False, 
                                                          target_size = (image_size, image_size),
                                                          class_mode = 'binary')

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

EPOCHS = 10

history = model.fit_generator(
    train_data,
    steps_per_epoch = int(np.ceil(60000/BATCH_SIZE)),
    epochs = EPOCHS,
    validation_data = validation_data,
    validation_steps = int(np.ceil(2300/BATCH_SIZE))
)


#Test the Model using new image
image_path="download.jpg"
img = image.load_img(image_path, target_size=(150, 150))
plt.imshow(img)
IMAGE_SHAPE = 160
img = np.expand_dims(img, axis=0)
image = np.array(img)
result = model.predict([image])
print(result)
plt.show()
