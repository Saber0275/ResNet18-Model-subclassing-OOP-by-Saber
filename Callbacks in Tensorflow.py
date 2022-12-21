'''Examples of Keras callback applications by Saber:
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import io
from PIL import Image

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, ReduceLROnPlateau


import os
import matplotlib.pylab as plt
import numpy as np
import math
import datetime
import pandas as pd
!rm -rf logs
%tensorboard --logdir logs

print("Version: ", tf.__version__)
tf.get_logger().setLevel('INFO')


#___________________________________________________________________________
# Download and prepare the horses or humans dataset

# horses_or_humans 3.0.0 has already been downloaded for you
path = "./tensorflow_datasets"
splits, info = tfds.load('horses_or_humans', data_dir=path, as_supervised=True, with_info=True, split=['train[:80%]', 'train[80%:]', 'test'])

(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

##########################################
print(f'N_examples:{num_examples}   and   N_Classes: {num_classes}')
##################################
SIZE = 150  #from 300*300
IMAGE_SIZE = (SIZE, SIZE)
##################################

def format_image(image, label): #define a function with 2 arguments(containing 2 types of info)
  image = tf.image.resize(image, IMAGE_SIZE) / 255.0  #size of pixel values between 0 and 1
  return  image, label

#_________________________________________________________


BATCH_SIZE = 32


train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1) #shuffle the samples in dataset
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)



#_______________________________________________________________________________

for image_batch, label_batch in train_batches.take(1):
  pass

image_batch.shape




'''Building Model:
'''
def build_model(dense_units, input_shape=IMAGE_SIZE + (3,)): #define a function with default/fixed arguments to generalize it later.
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(dense_units, activation='relu'),
      tf.keras.layers.Dense(2, activation='softmax')
  ])
  return model


#optimizing and loss:
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])



#________________________________________________________________________
'''1)
'''
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

model.fit(train_batches, 
          epochs=10, 
          validation_data=validation_batches, 
          callbacks=[tensorboard_callback])


#_______________________________________________________

'''Callback to save the Keras model or model weights at some frequency.'''
'''2)
'''
model.fit(train_batches, 
          epochs=5, 
          validation_data=validation_batches, 
          verbose=2,
          callbacks=[ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1),
          ]) #save the weights of the last 2 digits
##############
'''What is :.2f ?
'''
#Here we specify 2 digits of precision and f is used to represent floating point number.
"Floating point {:.2f}".format(345.7916732) 
##################

'''3)
'''
model.fit(train_batches, 
          epochs=1, 
          validation_data=validation_batches, 
          verbose=2,
          callbacks=[ModelCheckpoint('saved_model', verbose=1)
          ]) #saving model to model.h5 for each epoch


'''4)
'''
model.fit(train_batches, 
          epochs=2, 
          validation_data=validation_batches, 
          verbose=2,
          callbacks=[ModelCheckpoint('model.h5', verbose=1)
          ])





#EarlyStopping
'''5) Stop training when a monitored metric has stopped improving:
'''

model.fit(train_batches, 
          epochs=50, 
          validation_data=validation_batches, 
          verbose=2,
          callbacks=[EarlyStopping(
              patience=3,
              min_delta=0.05,
              baseline=0.8,
              mode='min',
              monitor='val_loss',
              restore_best_weights=True,
              verbose=1)
          ])






'''6) as CSV:
'''
  
csv_file = 'training.csv'

model.fit(train_batches, 
          epochs=5, 
          validation_data=validation_batches, 
          callbacks=[CSVLogger(csv_file)
          ])
csv_model=pd.read_csv(csv_file).head()





'''7) LearningRateScheduler →Updates the learning rate during training.
'''
  
def step_decay(epoch):
	initial_lr = 0.01
	drop = 0.5
	epochs_drop = 1
	lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lr

model.fit(train_batches, 
          epochs=5, 
          validation_data=validation_batches, 
          callbacks=[LearningRateScheduler(step_decay, verbose=1),
                    TensorBoard(log_dir='./log_dir')])



'''Reduce learning rate when a metric has stopped improving.
'''

'''8) ReduceLROnPlateau
'''
model = build_model(dense_units=256)
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])
  
model.fit(train_batches, 
          epochs=50, 
          validation_data=validation_batches, 
          callbacks=[ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.2, verbose=1,
                                       patience=1, min_lr=0.001),
                     TensorBoard(log_dir='./log_dir')])
#patience► be patient till the epoch!






























