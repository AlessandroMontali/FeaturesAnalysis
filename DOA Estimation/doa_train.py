import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from doa_data_provider import create_doa_dataset
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Create train and test datasets
filename = '/nas/home/amontali/data_doa/doa.tfrecord'
dataset = create_doa_dataset(filepath=filename, buffer_size=24480)
train_dataset = dataset.take(20000).batch(100).repeat()
test_dataset = dataset.skip(20000).batch(100).repeat()

# Create model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(4, 129, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
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
checkpoint_filepath = '/nas/home/amontali/models_doa/doa'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                               save_best_only=True, monitor='val_accuracy', mode='max',
                                                               verbose=1)

# Train model
history = model.fit(train_dataset, epochs=110, steps_per_epoch=200, validation_data=test_dataset, validation_steps=50,
                    callbacks=[model_checkpoint_callback])

# Plot loss and validation loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Train", "Test"])
plt.show()

# Plot accuracy and validation accuracy
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train", "Test"])
plt.show()


'''
# Fine tuning

checkpoint_filepath = '/nas/home/amontali/models_doa/doa'
model = tf.keras.models.load_model(checkpoint_filepath)

# Create train and test datasets

filename = '/nas/home/amontali/data_doa/doa_speech.tfrecord'
dataset = create_doa_dataset(filepath=filename, buffer_size=24480)
train_dataset = dataset.take(20000).batch(100).repeat()
test_dataset = dataset.skip(20000).batch(100).repeat()

# Evaluate model
model.evaluate(test_dataset, steps=50)

# Set number of epochs
initial_epochs = 110
fine_tune_epochs = 100
total_epochs = initial_epochs + fine_tune_epochs

# Save model
checkpoint_filepath_fine_tuning = '/nas/home/amontali/models_doa/doa_fine_tuning_speech'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_fine_tuning,
                                                               save_weights_only=False, save_best_only=True,
                                                               monitor='val_accuracy', mode='max', verbose=1)

# Train model
history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=initial_epochs, steps_per_epoch=200,
                         validation_data=test_dataset, validation_steps=50, callbacks=[model_checkpoint_callback])

# Plot loss and validation loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history_fine.history['loss'])
plt.plot(history_fine.history['val_loss'])
plt.legend(["Train", "Test"])
plt.show()

# Plot accuracy and validation accuracy
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(history_fine.history['accuracy'])
plt.plot(history_fine.history['val_accuracy'])
plt.legend(["Train", "Test"])
plt.show()
'''
