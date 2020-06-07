import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd

# Data preprocess
path_cats_and_dogs = getcwd() + "/cats-and-dogs.zip"
if not os.path.exists(getcwd() + '/tmp'):
    os.mkdir(getcwd() + '/tmp')
else:
    shutil.rmtree(getcwd() + '/tmp/')
    os.mkdir(getcwd() + '/tmp')

local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(getcwd()+'/tmp')
zip_ref.close()

print(len(os.listdir(getcwd() + '/tmp/PetImages/Cat/')))
print(len(os.listdir(getcwd() + '/tmp/PetImages/Dog/')))

try:
    dir1 = getcwd() + '/tmp/cats-v-dogs'
    dir2 = dir1 + '/training'
    dir3 = dir1 + '/testing'
    dir4 = dir2 + '/cats'
    dir5 = dir2 + '/dogs'
    dir6 = dir3 + '/cats'
    dir7 = dir3 + '/dogs'
    for i in range(7):
        os.mkdir(locals()['dir'+str(i+1)])
except OSError:
    pass

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    dataset = []
    all_data = os.listdir(getcwd()+SOURCE)
    for one_data in all_data:
        data_path = getcwd() + SOURCE + one_data
        if os.path.getsize(data_path) > 0:
            dataset.append(one_data)
        else:
            print('Skip {0} : Invalid file size'.format(one_data))
    
    training_data_length = int(len(all_data)*SPLIT_SIZE)
    testing_data_length = len(all_data) - training_data_length
    shuffled_data = random.sample(dataset, len(dataset))
    training_data = shuffled_data[:training_data_length]
    testing_data = shuffled_data[-testing_data_length:]

    for oneData in training_data:
        origin_path = getcwd() + SOURCE + oneData
        destination_path = getcwd() + TRAINING + oneData
        copyfile(origin_path, destination_path)
    for oneData in testing_data:
        origin_path = getcwd() + SOURCE + oneData
        destination_path = getcwd() + TESTING + oneData
        copyfile(origin_path, destination_path)
    
CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print("Training Cats Number: ", len(os.listdir(getcwd()+TRAINING_CATS_DIR)))
print("Training Dogs Number: ", len(os.listdir(getcwd()+TRAINING_DOGS_DIR)))
print("Testing Cats Number: ", len(os.listdir(getcwd()+TESTING_CATS_DIR)))
print("Testing Dogs Number: ", len(os.listdir(getcwd()+TESTING_DOGS_DIR)))
    
    

# build nn
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

TRAINING_DIR = getcwd() + "/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

VALIDATION_DIR = getcwd() + "/tmp/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)


# train
history = model.fit_generator(train_generator,
    steps_per_epoch=135,  # steps_per_epoch*batch_size = image_number
    epochs=40,
    verbose=1,
    validation_data = validation_generator,
    validation_steps=15)


# plot
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.figure()
plt.subplot(121)
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.title('Training and Validation Accuracy')
plt.legend()


plt.subplot(122)
plt.plot(epochs, loss, 'r', label='Training Loss' )
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')

plt.legend()
plt.show()