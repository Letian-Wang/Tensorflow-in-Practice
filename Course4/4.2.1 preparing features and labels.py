import tensorflow as tf

print("dataset = tf.data.Dataset.range(10)")
dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())

print("dataset = dataset.window(5, shift=1, drop_remainder=True)")
dataset = dataset.window(5, shift=1, drop_remainder=True)        
# window_size, drop_remainder(truncate data less than window_size)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()

print("dataset = dataset.flat_map(lambda window: window.batch(5))")
dataset = dataset.flat_map(lambda window: window.batch(5))      # turn data into numpy
for window in dataset:
    print(window.numpy())

print("dataset = dataset.map(lambda window: (window[:-1], window[-1:]))")
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))    # split feature and label
for x,y in dataset:
    print(x.numpy(), y.numpy())

print("dataset = dataset.shuffle(buffer_size=10)")
dataset = dataset.shuffle(buffer_size=10)      # shuffle data, buffer_size: amount of data we have
for x,y in dataset:
    print(x.numpy(), y.numpy())

print("dataset = dataset.batch(2).prefetch(1)")
dataset = dataset.batch(2).prefetch(1)          # divide into batch
for x,y in dataset:
    print("x = ", x.numpy())
    print("y = ",y.numpy())
