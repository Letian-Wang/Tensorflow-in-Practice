import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import shutil


# load model and load weights into model
path_inception = getcwd() + '/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# URL: https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels
local_weights_file = path_inception
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),  # desired shape
                                include_top = False,        # not include the DNN layer on the top of InceptionV3
                                weights = None              # not load weights
                                )           

pre_trained_model.load_weights(local_weights_file)          # load weights into network         
# pre_trained_model.summary()              

for layer in pre_trained_model.layers:
    layers.trainable = False                        # lock layers

# extend model
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)
model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

# callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.97):
            print('\nReached 97 accuracy, so cancelling training')
            self.model.stop_training = True

path_horse_or_human = getcwd() + '/horse-or-human.zip'
path_validation_horse_or_human = getcwd() + '/validation-horse-or-human.zip'


# extract data
if os.path.exists(getcwd()+'/tmp'):
    shutil.rmtree(getcwd()+'/tmp')
    os.mkdir(getcwd()+'/tmp')
else:
    os.mkdir(getcwd()+'/tmp')

local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()
local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()

# check directory
train_dir = getcwd() + '/tmp/training'
validation_dir = getcwd() + '/tmp/validation'
train_horses_dir = '/tmp/training/horses'
train_humans_dir = '/tmp/training/humans'
validation_horses_dir = '/tmp/validation/horses'
validation_humans_dir = '/tmp/validation/humans'

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print("length of train_horses_fnames: ", len(train_horses_fnames))
print("length of train_humans_fnames: ", len(train_humans_fnames))
print("length of validation_horses_fnames: ", len(validation_horses_fnames))
print("length of validation_humans_fnames: ", len(validation_humans_fnames))

# generate data
train_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size = 20,
    class_mode = 'binary'
)


# Train
callbacks = myCallback()
history = model.fit_generator(
    train_generator, 
    steps_per_epoch = 20,
    epochs = 100,
    callbacks=[callbacks],
    validation_data = validation_generator,
    validation_steps=50,
    verbose=1
)


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.subplot(121)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.subplot(121)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.savefig('2.2.1-transfer_learning.png')



