import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from doa_data_provider import create_doa_dataset
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='DOA training')
parser.add_argument('--filename_train_dataset', type=str, help='filename_train_dataset', default='/nas/home/amontali')
parser.add_argument('--filename_val_dataset', type=str, help='filename_val_dataset', default='/nas/home/amontali')
parser.add_argument('--checkpoint_filepath', type=str, help='checkpoint_filepath', default='/nas/home/amontali')
parser.add_argument('--save_filepath', type=str, help='save_filepath', default='/nas/home/amontali')
parser.add_argument('--gpu', type=str, help='gpu', default='0')
args = parser.parse_args()
filename_train_dataset = args.filename_train_dataset
filename_val_dataset = args.filename_val_dataset
checkpoint_filepath = args.checkpoint_filepath
save_filepath = args.save_filepath
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

'''
# Variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
filename_train_dataset = '/nas/home/amontali/data_doa/speech_T60_0.2_SNR_20_train2.tfrecord'
filename_val_dataset = '/nas/home/amontali/data_doa/speech_T60_0.2_SNR_20_val2.tfrecord'
checkpoint_filepath = '/nas/home/amontali/models_doa/T60_0.2_SNR_20_new2v3'
save_filepath = '/nas/home/amontali/tmp/T60_0.2_SNR_20_acc2v3.png'
'''

# Create train and test datasets
train_dataset = create_doa_dataset(filepath=filename_train_dataset, buffer_size=1200)
train_dataset = train_dataset.repeat().batch(100)
val_dataset = create_doa_dataset(filepath=filename_val_dataset, buffer_size=300)
val_dataset = val_dataset.repeat().batch(100)

# Create model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(8, 129, 1)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(37, activation='softmax'))

# Visualize model
model.summary()

# Compile model
model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

# Save model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                               save_best_only=True, monitor='val_accuracy', mode='max',
                                                               verbose=1)

# Train model
history = model.fit(train_dataset, epochs=350, steps_per_epoch=12, validation_data=val_dataset, validation_steps=3,
                    callbacks=[model_checkpoint_callback])

# Plot accuracy and validation accuracy
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train", "Test"])
plt.savefig(save_filepath)
plt.show()

'''
# Fine tuning

checkpoint_filepath = '/nas/home/amontali/models_doa/T60_0.2_SNR_20_new2v3'
model = tf.keras.models.load_model(checkpoint_filepath)
# Create train and test datasets

filename_train_dataset = '/nas/home/amontali/data_doa/speech_T60_0.2_SNR_20_train2.tfrecord'
filename_val_dataset = '/nas/home/amontali/data_doa/speech_T60_0.2_SNR_20_val2.tfrecord'
train_dataset = create_doa_dataset(filepath=filename_train_dataset, buffer_size=1200)
train_dataset = train_dataset.repeat().batch(100)
val_dataset = create_doa_dataset(filepath=filename_val_dataset, buffer_size=300)
eval_dataset = val_dataset.repeat().batch(100)

# Evaluate model
model.evaluate(eval_dataset, steps=3)

# Set number of epochs
initial_epochs = 350
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

# Save model
checkpoint_filepath_fine_tuning = '/nas/home/amontali/models_doa/T60_0.2_SNR_20_new2v3_ft'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_fine_tuning,
                                                               save_weights_only=False, save_best_only=True,
                                                               monitor='val_accuracy', mode='max', verbose=1)

# Train model
history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=initial_epochs, steps_per_epoch=12,
                         validation_data=eval_dataset, validation_steps=3, callbacks=[model_checkpoint_callback])


# Compute MAE

def average(lst):
    return sum(lst) / len(lst)


DOAs = np.arange(0, 181, 5)
deg_errors = []
for data in val_dataset:
    prediction = model.predict(tf.expand_dims(data[0], axis=0))
    deg_errors.append(mean_absolute_error([DOAs[np.argmax(data[1])]], [DOAs[np.argmax(prediction[0])]]))
mae = average(deg_errors)
print(mae)
'''
