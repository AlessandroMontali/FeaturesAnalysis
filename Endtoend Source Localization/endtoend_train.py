import tensorflow as tf
import matplotlib.pyplot as plt
from endtoend_data_provider import create_endtoend_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.losses import mean_squared_error
import argparse
import os

parser = argparse.ArgumentParser(description='Endtoend training')
parser.add_argument('--filename_train_dataset', type=str, help='filename_train_dataset', default='/nas/home/amontali/data_entoend')
parser.add_argument('--filename_val_dataset', type=str, help='filename_val_dataset', default='/nas/home/amontali/data_entoend')
parser.add_argument('--checkpoint_filepath', type=str, help='checkpoint_filepath', default='/nas/home/amontali/model_entoend')
parser.add_argument('--save_filepath', type=str, help='save_filepath', default='/nas/home/amontali/save_entoend')
parser.add_argument('--gpu', type=str, help='gpu', default='0')
args = parser.parse_args()
filename_train_dataset = args.filename_train_dataset
filename_val_dataset = args.filename_val_dataset
checkpoint_filepath = args.checkpoint_filepath
save_filepath = args.save_filepath
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Create train and test datasets
train_dataset = create_endtoend_dataset(filepath=filename_train_dataset, buffer_size=1200)
train_dataset = train_dataset.repeat().batch(100)
val_dataset = create_endtoend_dataset(filepath=filename_val_dataset, buffer_size=300)
val_dataset = val_dataset.repeat().batch(100)

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
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                               save_best_only=True, monitor='val_loss', mode='min',
                                                               verbose=1)

# Train model
history = model.fit(train_dataset, epochs=150, steps_per_epoch=12, validation_data=val_dataset,
                    validation_steps=3, callbacks=[model_checkpoint_callback])

# Plot loss and validation loss
plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Train", "Validation"])
plt.savefig(save_filepath)
plt.show()
