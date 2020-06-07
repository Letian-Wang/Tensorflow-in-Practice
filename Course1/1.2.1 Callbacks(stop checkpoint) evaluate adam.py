import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.2):   # 'accuracy'
            print("\nReached 60% accuracy so cancelling training!")
            # model.save('tmp.h5')
            self.model.stop_training = True

mc = tf.keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=5)

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images/255.0
test_images=test_images/255.0
print("training_images: ", training_images[0])
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=20, callbacks=[mc, callbacks])
print("evaluate")
model.evaluate(test_images, test_labels)
print('predict')
classifications = model.predict(test_images)
print(classifications[0])
