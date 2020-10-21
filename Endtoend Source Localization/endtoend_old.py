import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.losses import mean_squared_error
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(feature0,feature1):
    feature = {
       'signal': _bytes_feature(feature0),
       'source': _bytes_feature(feature1),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def read_tfrecord(serialized_example):
    feature_description = {
        'source': tf.io.FixedLenFeature((), tf.string),
        'microphone': tf.io.FixedLenFeature((), tf.string),
        'signals': tf.io.FixedLenFeature((), tf.string),
        'win_signals_1': tf.io.FixedLenFeature((), tf.string),
        'win_signals_2': tf.io.FixedLenFeature((), tf.string),
        'win_signals_3': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    y_train = tf.io.parse_tensor(example['source'], out_type = tf.float64)
    x_train = tf.io.parse_tensor(example['win_signals_1'], out_type = tf.float64)
    return x_train, y_train


filename = '/nas/home/amontali/endtoend_data.tfrecord'
tfrecord_dataset = tf.data.TFRecordDataset(filename)
dataset = tfrecord_dataset.map(read_tfrecord)

signals_input_dataset = []
sources_input_dataset = []
for data in dataset:
    slice_idx = tf.random.uniform(shape=(), minval=0, maxval=data[0].shape[2]-1, dtype=tf.dtypes.int64)
    matrix_input = tf.slice(data[0],begin=[0, 0, slice_idx],size = [8, 1280, 1])
    matrix_input = tf.math.divide(tf.subtract(matrix_input, tf.reduce_min(matrix_input)), tf.subtract(tf.reduce_max(matrix_input), tf.reduce_min(matrix_input)))
    matrix_input = tf.squeeze(matrix_input,axis=2)
    matrix_input = tf.reshape(matrix_input, [matrix_input.shape[1], matrix_input.shape[0]])
    signals_input_dataset.append(matrix_input)
    sources_input_dataset.append(data[1])
dataset = tf.data.Dataset.from_tensor_slices((signals_input_dataset,sources_input_dataset))

dataset = dataset.shuffle(2262)
train_size = 1700 #1700 train & 562 test
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
'''
# Save file with test data
filename_test_data = '/nas/home/amontali/endtoend_model_old.tfrecord'
with tf.io.TFRecordWriter(filename_test_data) as writer:
    for elem in test_dataset:
        feature0 = np.expand_dims(elem[0], axis=0)
        feature1 = np.expand_dims(elem[1], axis=0)
        dataset_test = tf.data.Dataset.from_tensor_slices((feature0, feature1))
        for feature0, feature1 in dataset_test:
            example = serialize_example(tf.io.serialize_tensor(feature0), tf.io.serialize_tensor(feature1))
            writer.write(example)
'''
train_dataset = train_dataset.batch(100).repeat()
test_dataset = test_dataset.batch(100).repeat()

model = Sequential()

model.add(Conv1D(filters=96, kernel_size=7, padding='same', activation='relu', input_shape=(1280, 8)))
model.add(MaxPooling1D(pool_size=7))
model.add(Conv1D(filters=96, kernel_size=7, padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(3, activation='linear'))

model.summary()

model.compile(loss=mean_squared_error, optimizer='adam', metrics=['accuracy'])
'''
checkpoint_filepath = '/nas/home/amontali/model_endtoend_old'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                               save_best_only=True, monitor='val_loss', mode='min',
                                                               verbose=1)
'''
history = model.fit(train_dataset, epochs=200, steps_per_epoch=17, validation_data=test_dataset,
                    validation_steps=6)

model.evaluate(test_dataset, steps=6)

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Train", "Test"])
plt.show()

