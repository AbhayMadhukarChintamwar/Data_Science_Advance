from Pathlib import TRAIN_DIR, TEST_DIR, PREDICT_DIR, MODEL_DIR
import numpy as np
import pandas as pd
import tensorflow as tf
import keras 
import warnings
warnings.simplefilter('ignore')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image



gen = ImageDataGenerator(rescale = 1/255, shear_range = 0.2, zoom_range= 0.2)
training_set = gen.flow_from_directory(TRAIN_DIR, target_size = (64, 64), class_mode = 'binary')
print(training_set.class_indices)

test = ImageDataGenerator(rescale = 1/255)
test_set = test.flow_from_directory(TEST_DIR, target_size = (64,64), class_mode = 'binary')
print(test_set.class_indices)


cnn = Sequential()

cnn.add(Conv2D(filters =32, kernel_size = 3, activation='relu', input_shape =[64,64,3]))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(Flatten())


cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy'])
cnn.fit(x = training_set, validation_data=test_set, epochs=25)


test_image = Image.open(PREDICT_DIR/'dog.21.jpg')
test_image = test_image.resize((64,64))
test_image = np.array(test_image)
test_image = np.expand_dims(test_image,axis = 0)

result = cnn.predict(test_image)
print(result)

if result[0][0]==1:
    print('dog')
else:
    print('cat')


