# Simple RNN
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),    # input_shape: [0] is batch_size, 1 is timestamp
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])

# LSTM
model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])

# clear cache
tf.keras.backend.clear_session()

# Lambda
# input of RNN: [batch_size, time_stamps, series_dimension]
model = keras.models.Sequential([
    keras.layers.Lambda(lambda x:tf.expand_dims(x, axis=-1), input_shape=[None]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x:x*100)
])

# huber loss and lr-tuning
# huber: less sensitive to outliers
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch/20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss = tf.keras.lossed.Huber(), optimizer=optimizer, metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])



