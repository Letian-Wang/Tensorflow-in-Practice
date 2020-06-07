import tensorflow as tf
from os import getcwd
vocab_size = 1000
embedding_dim = 16
max_length = 120
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.load_weights(getcwd()+'/test.h5')


e = model.layers[0]
weights = e.get_weights()[0]
print(type(weights))
print(weights.shape)
print(weights)

