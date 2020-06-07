import tensorflow as tf
from os import path, getcwd, chdir
import os
import matplotlib.pyplot as plt

cur_dir = os.getcwd()

path = cur_dir + "/mnist.npz"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
def train_mnist_conv():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99.8 accuracy so cancelling training")
                self.model.stop_training = True
    callbacks = myCallback()

    # Get data
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # Process data for CNN
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255 
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255
    


    # Build nn
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])

    # Train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
    classification = model.predict(test_images)
    print("classification[0]: ", classification[0])
    model.evaluate(test_images, test_labels)
    return history.epoch, history.history['accuracy'][-1]

_,_ = train_mnist_conv()
