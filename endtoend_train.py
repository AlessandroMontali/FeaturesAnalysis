import tensorflow as tf
import matplotlib.pyplot as plt
from endtoend_data_provider import create_endtoend_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.losses import mean_squared_error
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# Create train and test datasets

filename = '/nas/home/amontali/endtoend_data.tfrecord'
train_dataset = create_endtoend_dataset(filepath=filename, buffer_size=2262, batch_size=100, train_size=1700,
                                        dataset_type='train')
test_dataset = create_endtoend_dataset(filepath=filename, buffer_size=2262, batch_size=100, train_size=1700,
                                       dataset_type='test')

# Create model

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

# Visualize model
model.summary()

# Compile model
model.compile(loss=mean_squared_error, optimizer='adam', metrics=['accuracy'])

# Save model
checkpoint_filepath = '/nas/home/amontali/model_endtoend'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                               save_best_only=True, monitor='val_loss', mode='min',
                                                               verbose=1)

# Train model
history = model.fit(train_dataset, epochs=1000, steps_per_epoch=17, validation_data=test_dataset,
                    validation_steps=6, callbacks=[model_checkpoint_callback])

# Plot loss and validation loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Train", "Test"])
plt.show()

'''
# Fine tuning

checkpoint_filepath = '/nas/home/amontali/model_endtoend'
model = tf.keras.models.load_model(checkpoint_filepath)

# Create train and test datasets

filename = '/nas/home/amontali/endtoend_data.tfrecord'
train_dataset = create_endtoend_dataset(filepath=filename, buffer_size=2262, batch_size=100, train_size=1700,
                                        dataset_type='train')
test_dataset = create_endtoend_dataset(filepath=filename, buffer_size=2262, batch_size=100, train_size=1700,
                                       dataset_type='test')

# Evaluate model
model.evaluate(test_dataset, steps=6)

# Set number of epochs
initial_epochs = 1000
fine_tune_epochs = 1000
total_epochs = initial_epochs + fine_tune_epochs

# Save model
checkpoint_filepath_fine_tuning = '/nas/home/amontali/tmp/model_endtoend_fine_tuning'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_fine_tuning,
                                                               save_weights_only=False, save_best_only=True,
                                                               monitor='val_loss', mode='min', verbose=1)

# Train model
history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=initial_epochs, steps_per_epoch=17,
                         validation_data=test_dataset, validation_steps=6, callbacks=[model_checkpoint_callback])

# Plot loss and validation loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history_fine.history['loss'])
plt.plot(history_fine.history['val_loss'])
plt.legend(["Train", "Test"])
plt.show()
'''