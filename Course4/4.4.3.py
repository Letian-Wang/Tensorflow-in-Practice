import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from os import getcwd

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
def preprocess():
    def source_data():
        temps = []
        with open(getcwd()+'/daily-min-temperatures.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                temps.append(float(row[1]))
        time_step = [i+1 for i, key in enumerate(temps)]
        series = np.array(temps)
        time = np.array(time_step)
        plt.figure('source data', figsize=(10, 6))
        plot_series(time, series)
        return time,series
    def prepare_feature_and_label(time, series):
        def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
            series = tf.expand_dims(series, axis=-1)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size + 1))
            ds = ds.shuffle(shuffle_buffer)
            ds = ds.map(lambda w: (w[:-1], w[1:]))
            return ds.batch(batch_size).prefetch(1)
        

        time_train = time[:split_time]
        x_train = series[:split_time]
        time_valid = time[split_time:]
        x_valid = series[split_time:]

        tf.keras.backend.clear_session()
        tf.random.set_seed(54)
        np.random.seed(51)
       
        train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
        return train_set, series, time_valid, x_valid

    time, series = source_data()
    train_set, series, time_valid, x_valid = prepare_feature_and_label(time, series)
    
    return train_set, series, time_valid, x_valid
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu', input_shape=[None,1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        # tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    return model
def lr_tuning(model, train_set):
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8*10**(epoch/4))
    optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
    history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
    plt.semilogx(history.history["lr"], history.history["loss"])
    plt.axis([1e-8, 1e-3, 0, 60])
    return history
def train(model, train_set):
    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    # model.load_weights('80.h5')
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    mc = tf.keras.callbacks.ModelCheckpoint('/save/weights{:08d}.h5', save_weights_only=True, period=10)
    history = model.fit(train_set,epochs=720, callbacks=[mc])
    # model.save('800.h5')
    loss = history.history['loss']
    mae = history.history['mae']
    epochs = range(len(loss))
    plt.figure('Training Loss')
    plt.plot(epochs, loss)
    plt.plot(epochs, mae)
    plt.title('Training Loss')
    plt.legend(['Loss', 'Mae'])
    return history
def forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w:w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

split_time = 2500
window_size = 64       # 64
batch_size = 256
shuffle_buffer_size = 1000
train_set, series, time_valid, x_valid = preprocess()
model = build_model()
# history = lr_tuning(model, train_set)
history = train(model, train_set)
forecast = forecast(model, series[...,np.newaxis], window_size)
forecast = forecast[split_time-window_size:-1, -1, 0]
print("mae: ", tf.keras.metrics.mean_absolute_error(x_valid, forecast).numpy())
plt.figure('forecast', figsize=(10,6))
plot_series(time_valid, x_valid)
plot_series(time_valid, forecast)
plt.legend(['groundtruth', 'forecast'])
plt.show()
