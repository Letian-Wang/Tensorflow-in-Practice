import tensorflow as tf
print(tf.__version__)
# tf.enable_eager_execution()       # not need in tf 2.0
# !pip install -q tensorflow-datasets

# Get data
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
import numpy as np
train_data, test_data = imdb['train'], imdb['test']         # 25000 training, 25000 testing
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
for s,l in train_data:
    training_sentences.append(str(s.numpy()))     # convert tensor to numpy
    training_labels.append(l.numpy())
for s,l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# Preprocess data
vocab_size = 10000          # not setting it means the number equals the number of the datas
embedding_dim = 16
max_length = 120            # default: length of longest texts
padding = "post"            # padding from post, default: fore
trunc_type = 'post'         # lost infomation of texts from post, default: fore
oov_tok = "<OOV>"               # Out of vocabulary
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)          
tokenizer.fit_on_texts(training_sentences)                          # create dictionary
word_index = tokenizer.word_index
# Transfer text to sequences
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

# Reverse
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(padded[3]))
print(training_sentences[3])

# build nn
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.GlobalAveragePooling1D(),         # another flatten, smaller dimension, faster
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs = num_epochs, validation_data = (testing_padded, testing_labels_final))

# look at layer
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)       # shape:(vocab_size, embedding_dim)


sentence = "I really think this is amazing. honest."
sequence = tokenizer.texts_to_sequences(sentence)
print(sequence)