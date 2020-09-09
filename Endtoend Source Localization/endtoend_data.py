import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# Generate data and save TFRecord file

filename = '/nas/home/amontali/data_endtoend/T60_0.5_SNR_40.tfrecord'

# Specify reverberation and noise
T60 = 0.5
SNR = 40


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(feature0,feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8):
    feature = {
       'source': _bytes_feature(feature0),
       'microphone': _bytes_feature(feature1),
       'signals': _bytes_feature(feature2),
       'win_signals_1': _bytes_feature(feature3),
       'win_signals_2': _bytes_feature(feature4),
       'win_signals_3': _bytes_feature(feature5),
       'win_signals_1_shape': _bytes_feature(feature6),
       'win_signals_2_shape': _bytes_feature(feature7),
       'win_signals_3_shape': _bytes_feature(feature8),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Download dataset
corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl'])

# Specify room dimensions
corners = np.array([[0, 0], [0, 8.2], [3.6, 8.2], [3.6, 0]]).T  # [x,y]

# Sources generation [Room subdivision: 14*28*3 for 1176 elements or 29*60*9 for 15660 elements]
rows_number = 14
cols_number = 28
depths_number = 3

rows = np.arange(0, 3.6, 3.6/rows_number)
cols = np.arange(0, 8.2, 8.2/cols_number)
depths = np.arange(0.92, 1.53, (1.53-0.92)/depths_number)

sources_pos=[]
for i in range(rows_number):
    for j in range(cols_number):
        for k in range(depths_number):
            pos = [rows[i],cols[j],depths[k]]
            sources_pos.append(pos)

# Number of examples: 1131 (US bdl) - 15583 (entire database)
N = len(corpus)

# Windowing: 80ms,160ms,320ms -> 1280,2560,5120 samples
windows = [1280, 2560, 5120]

with tf.io.TFRecordWriter(filename) as writer:
    for j in range(0, N, 1):
        signal = corpus[j].data
        fs = corpus[j].fs
        time = np.arange(0, len(signal)) / fs

        # Add source to 3D room (set max_order to a low value for a quick, but less accurate, RIR)
        source_position = sources_pos[j]
        room = pra.Room.from_corners(corners, fs=fs, max_order=8, absorption=0.16/T60)
        room.extrude(2.4)
        room.add_source(source_position, signal=signal)

        # Add microphones to 3D room
        mic_center_1 = np.array([1.8, 4.5, 1.2])
        mic_center_2 = np.array([1.8, 3.7, 1.2])
        mic_centers = [mic_center_1, mic_center_2]

        for i in range(2):
            R = pra.circular_2D_array(mic_centers[i][:2], M=8, phi0=0, radius=0.1)
            R = np.concatenate((R, np.ones((1, 8)) * mic_centers[i][2]), axis=0)
            mics = pra.MicrophoneArray(R, room.fs)
            room.add_microphone_array(mics)

            # Compute image sources
            room.image_source_model()

            # Simulate the propagation
            room.compute_rir()
            room.simulate(snr=SNR)

            # Plot the results
            # room.plot_rir()
            # fig = plt.gcf()
            # fig.set_size_inches(20, 15)

            print("MICROPHONES SIGNAL SHAPE FOR SOURCE", j + 1, "AND MICROPHONE", i + 1)
            print("Signal shape for every M:", room.mic_array.signals.shape)

            # Apply windows to the signal
            for w in range(3):
                window_size = windows[w]
                win = tf.signal.hann_window(window_size, dtype=tf.dtypes.float64)
                frames = tf.signal.frame(room.mic_array.signals, window_size, window_size)
                win_sig = win * frames
                win_sig = tf.reshape(win_sig, [win_sig.shape[0], win_sig.shape[2], win_sig.shape[1]])
                if w == 0:
                    w1 = win_sig
                if w == 1:
                    w2 = win_sig
                if w == 2:
                    w3 = win_sig
                print("Windowed signal shape for each M:", win_sig.shape)

            # Create dataset amd write TFRecord file
            feature0 = np.expand_dims(sources_pos[j], axis=0)
            feature1 = np.expand_dims(mic_centers[i], axis=0)
            feature2 = np.expand_dims(room.mic_array.signals, axis=0)
            feature3 = np.expand_dims(w1, axis=0)
            feature4 = np.expand_dims(w2, axis=0)
            feature5 = np.expand_dims(w3, axis=0)
            feature6 = np.expand_dims(w1.shape, axis=0)
            feature7 = np.expand_dims(w2.shape, axis=0)
            feature8 = np.expand_dims(w3.shape, axis=0)

            dataset = tf.data.Dataset.from_tensor_slices(
                (feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8)
            )

            for feature0, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 in dataset:
                example = serialize_example(tf.io.serialize_tensor(feature0), tf.io.serialize_tensor(feature1),
                                            tf.io.serialize_tensor(feature2), tf.io.serialize_tensor(feature3),
                                            tf.io.serialize_tensor(feature4), tf.io.serialize_tensor(feature5),
                                            tf.io.serialize_tensor(feature6), tf.io.serialize_tensor(feature7),
                                            tf.io.serialize_tensor(feature8))
                writer.write(example)

            print("-------------------------------------------------------")
