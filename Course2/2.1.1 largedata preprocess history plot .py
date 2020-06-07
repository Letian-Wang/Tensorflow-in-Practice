import os 
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd

# extract data
path_cats_and_dogs = getcwd() + "/cats-and-dogs.zip"
if not os.path.exists(getcwd() + '/tmp'):
    os.mkdir(getcwd() + '/tmp')
else:
    shutil.rmtree(getcwd() + '/tmp')
local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall(getcwd()+'/tmp')
zip_ref.close()
print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))

# create dirs
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

# split training and testing data
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    data_set = []
    for oneData in os.listdir(SOURCE):
        data = SOURCE + oneData
        if os.path.getsize(data) > 0:
            data_set.append(oneData)
        else:
            print('Skip {0} : Invalid file size'.format(data))
    print("length of data_set: ", len(data_set))
    train_data_length = int(len(data_set) * SPLIT_SIZE)
    test_data_length = int(len(data_set) - train_data_length)
    print("train_data_length: ", train_data_length)
    print("test_data_length: ", test_data_length)
    shuffled_set = random.sample(data_set, len(data_set))
    training_data = shuffled_set[0:train_data_length]
    testing_data = shuffled_set[-test_data_length:]
    print("length of training_data: ", len(training_data))
    print("length of test_data_length: ", len(testing_data))


    for unitData in training_data:
        temp_train_data = SOURCE + unitData
        final_train_data = getcwd() + TRAINING + unitData
        copyfile(temp_train_data, final_train_data)
    for oneData in testing_data:
        origin_data = SOURCE + oneData
        destin_data = getcwd() + TESTING + oneData
        copyfile(origin_data, destin_data)
CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"
split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics = ['accuracy'])

# generate training data
TRAINING_DIR = getcwd() + "/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size = 10,
    class_mode ='binary'
)

VALIDATION_DIR = getcwd() + '/tmp/cats-v-dogs/testing/'
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=10,
    class_mode = 'binary'
)

history = model.fit(train_generator, epochs = 2, verbose = 1,
                                validation_data=validation_generator)


# plot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()


