import json
import tensorflow as tf
import csv
import random
import numpy as np
from os import getcwd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

embedding_dim = 100
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 16000
test_portion = 0.1

# Get data
corpus = []
num_sentences = 0
with open(getcwd()+"/training_cleaned.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        label = 0 if row[0]=='0' else 1
        text = row[5]
        list_item=[text, label]
        num_sentences = num_sentences + 1
        corpus.append(list_item)
print(num_sentences)
print(len(corpus))
print(corpus[1])

# tokenizer
sentences = []
labels = []
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index)
print("vocab_size: ", vocab_size)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_length = int(test_portion*training_size)
test_sequences = padded[0:testing_length]
training_sequences = padded[testing_length:training_size]
test_labels = labels[0:testing_length]
training_labels = labels[testing_length:training_size]

# Get pretrained first-layer weight 
embeddings_index = {}
with open(getcwd()+'/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # print("shape of embedding_vector: ", embedding_vector)
        # print("shape of embeddings_matrix[i]: ", embeddings_matrix[i])
        embeddings_matrix[i] = embedding_vector
print(len(embeddings_matrix))        

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    # tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_corssentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
num_epochs = 50
print(training_sequences[0])
history = model.fit(training_sequences, training_labels, epochs=num_epochs, validation_data=(test_sequences, test_labels), verbose=2)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.figure()
plt.plot(epochs, acc, label='training accuracy')
plt.plot(epochs, val_acc, label='testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, label='training loss')
plt.plot(epochs, val_loss, label='testing loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
