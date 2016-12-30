##THEANO_FLAGS='floatX=float32,device=cpu,lib.cnmem=1' python <myscript>.py

############## ALL IMPRTS ########################
from __future__ import print_function
import numpy as np
np.random.seed(1337)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD
from keras.callbacks import History, ModelCheckpoint, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

import pandas as pd

############## ALL USER FUNCTIONS ########################

def plot_progress(epoch,logs):
    plt.figure()
    plt.plot(range(epoch+1),history.history['loss'],'b',label='trainin loss')
    plt.plot(range(epoch+1),history.history['val_loss'],'r',label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('training error')
    plt.legend(loc='best')
    plt.savefig(locpath+'training_error.png')
    plt.close('all')

############## TRAINING SETTINGS ########################

#K.set_image_dim_ordering('th') # For image align

batch_size = 32
nb_epoch = 50
nb_classes = 8
im_shape = (320, 180)
im_shape_dim = (320, 180, 3)

locpath = "/Users/piris/Documents/Kaggle/Nature/trainingResults/"
filepath = locpath+"net.{epoch:02d}-{val_loss:.2f}.h5"
weights_path = "/Users/piris/Documents/Kaggle/Nature/weights_pretrained/vgg16_weights.h5"

############## LOAD TRAINING DATA ########################

train_datagen = ImageDataGenerator(
                                   rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                '../data/train',  # this is the target directory
                target_size=(im_shape),  # all images will be resized to 224x224
                batch_size=batch_size,
                class_mode='categorical')  # since there are 8 different fish labels

validation_generator = test_datagen.flow_from_directory(
                '../data/val',
                target_size=(im_shape),
                batch_size=batch_size,
                class_mode='categorical')

############## DEFINE MODEL ########################

# create the base pre-trained model
base_net = InceptionV3(weights='imagenet', include_top=False,input_shape=im_shape_dim)

# add a global spatial average pooling layer
x = base_net.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 8 classes
predictions = Dense(8, activation='softmax')(x)

# this is the model we will train
net = Model(input=base_net.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in net.layers[:210]:
    layer.trainable = False
for layer in net.layers[210:]:
    layer.trainable = True

#opt = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
#opt = Adadelta()
opt=SGD(lr=0.001, momentum=0.9)

net.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

history = History()
plot_progress_cb = LambdaCallback(on_epoch_end=plot_progress)

check = ModelCheckpoint(filepath=filepath,monitor='val_acc', verbose=1,
                        save_best_only=True)

print("Starting training...")

net.fit_generator(
            train_generator,
            samples_per_epoch=3390,
            nb_epoch=nb_epoch,
            verbose=1,
            callbacks=[history,check,plot_progress_cb],
            validation_data=validation_generator,
            nb_val_samples=357)


print("Training end...")
