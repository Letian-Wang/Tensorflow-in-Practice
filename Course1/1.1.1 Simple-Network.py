import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
# one layer, one input
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

history = model.fit(xs, ys, epochs=500)
loss = history.history['loss']
print("========")
print(model.predict([10.0]))
model.evaluate([10, 20], [20, 40])
epochs = [i for i in range(len(loss))]
plt.scatter(epochs, loss, label='loss')
plt.show()