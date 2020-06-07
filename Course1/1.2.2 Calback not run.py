import tensorflow as tf
from os import path, getcwd, chdir
def train_mnist():
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('loss')<0.4):
                print("\nReached 99 accuracy so canceling training!")
                self.model.stop_training = True
    callbacks = myCallback
    (x_train, y_train), (x_test, y_tesg) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])
    model.compile(optimization='sgd', loss='sparse_catogorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, callbacks = [callbacks])
    return history.epoch, history.history['acc'][-1]

 

