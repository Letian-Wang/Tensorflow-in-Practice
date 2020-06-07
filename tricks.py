# Basics
model.predict                   # give input, see output
model.evaluate                  # give input and label to see loss
model.summary()                 # see network structure
history = model.fit             # record for plot
callback                        # cancel training when a reaching a threshold
tf.keras.models.Sequential      # build nn
dropout()
test_loss, test_acc = model.evaluate(training_images, training_labels)

# optimizer and loss:
RMSprop, admam, sgd
binary_crossentropy
categorical_crossentropy

# data generator, augmentation: 
train_datagen = ImageDataGenerator(rescale = 1 /255, rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, zoom_range=0.2, shear_range=0.2, horizontal_flip = True, fill_mode = 'nearest')
train_datagen.flow(training_images, training_labels, batch_size=32)
train_datagen.flow_from_directory(VALIDATION_DIR, target_size=(150, 150), batch_size=10, class_mode='binary')
'''steps_per_epoch*batch_size = len(training_images)'''

# data auto label with dir name: 
train_generator = train_datagen.flow_from_directory()

# Transfer learning:
path_inception = getcwd() + '/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layers.trainable = False                        # lock layers

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
x = layers.Flatten()(last_output)

# Get certain layer
l0 = tf.keras.layers.Dense(1, input_shape = [window_size])
print("Layer weights {}".format(l0.get_weights()))


# set weight of certain layer
tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),

# weight, data are all in list
[[1,2,3],
 [4,5,6],
 [7,8,9]]

# one-hot encode
label = ku.to_categorical(label, num_classes=total_words)           



''' Callback to save model '''
''' Save Model '''
''' 1. Regular save '''
# model.save('/tmp/model')
# loaded_model = tf.keras.models.load_model('/tmp/model')
''' 2. Callback: constantly save the best model '''
# EPOCHS = 10
# checkpoint_filepath = '/tmp/checkpoint'
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_acc',
#     mode='max',
#     save_best_only=True)

# # Model weights are saved at the end of every epoch, if it's the best seen
# # so far.
# model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# # The model weights (that are considered the best) are loaded into the model.
# model.load_weights(checkpoint_filepath)

''' 3. Callback: save model every 5 epochs '''
# mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=5)
# model.fit(X_train, Y_train, callbacks=[mc])

# panda plot
df = pd.DataFrame(history.history)
print(df.head())

loss_plot = df.plot(y='loss', title="Loss vs. Epoch", legend=False)
loss_plot = df.plot(y='mean_absolute_error', title="Loss vs. Epoch", legend=False)
loss_plot.set(xlabel='Epochs', ylabel='Loss')

# input_shape
Dense(5, input_shape = (feature_dim))
x_train shape: (num_sample, (feature_dims))         # Conv need a channel_dim at the end
y_train shape: (num_sample, (label_dim))
layer shape: (batch_size, (feature_dims))