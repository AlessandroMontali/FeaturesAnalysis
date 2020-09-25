import numpy as np
import pyroomacoustics as pra
import tensorflow as tf
import math
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

data_type = 'speech'

# Generate data and save TFRecord file
if data_type == 'noise':
    filename = '/nas/home/amontali/data_doa/doa.tfrecord'
else:
    filename = '/nas/home/amontali/data_doa/doa_speech.tfrecord'

# Specify reverberation and noise
T60 = 0.2
SNR = 20


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3, feature4, feature5, feature6):
    feature = {
        'source': _bytes_feature(feature0),
        'microphone': _bytes_feature(feature1),
        'signals': _bytes_feature(feature2),
        'phase_signals': _bytes_feature(feature3),
        'phase_signals_shape': _bytes_feature(feature4),
        'doa': _bytes_feature(feature5),
        'doa_classes': _bytes_feature(feature6),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Specify room dimensions
corners = np.array([[0, 0], [0, 8.2], [3.6, 8.2], [3.6, 0]]).T  # [x,y]

# Specify microphone position and source-array distances
mic_center = np.array([1.8, 4.5, 0.0])
if data_type == 'noise':
    distances = [1, 2]
else:
    distances = [1.5]

# DOA classes
DOA_range = [0, 181]
resolution = 5
DOAs = np.arange(DOA_range[0], DOA_range[1], resolution)

# Specify angles for source generation ((30-150)*2 -> 240 sources for noise & (30-150) -> 120 for speech)
angles = np.arange(30, 150, 1)
N = len(angles) * len(distances)

# Sources generation
sources_pos = []
sources_doa = []
for dist in distances:
    for teta in angles:
        if teta > 90:
            teta_deg = 180 - teta
        else:
            teta_deg = teta
        teta_rad = (teta_deg * math.pi) / 180

        h_source = dist * math.sin(teta_rad)
        x_source = dist * math.cos(teta_rad)

        if teta > 90:
            x_source = mic_center[0] + x_source
        else:
            x_source = mic_center[0] - x_source

        source = [x_source, mic_center[1], h_source]
        sources_pos.append(source)
        sources_doa.append(teta)

# Noise signals generation
noise_signals = []
for i in range(N):
    noise = np.random.normal(0, 1, 10000)
    noise_signals.append(noise)

# Download speech dataset
corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl'])

with tf.io.TFRecordWriter(filename) as writer:
    for j in range(0, N, 1):
        if data_type == 'noise':
            signal = noise_signals[j]
            fs = 16000
        else:
            signal = corpus[j].data
            fs = corpus[j].fs

        # Add source to 3D room
        source_position = sources_pos[j]
        room = pra.Room.from_corners(corners, fs=fs, max_order=8, absorption=0.16 / T60)
        room.extrude(2.4)
        room.add_source(source_position, signal=signal)

        # Add microphones to 3D room
        R = pra.linear_2D_array(mic_center[:2], M=4, phi=0, d=0.03)
        R = np.concatenate((R, np.ones((1, 4)) * mic_center[2]), axis=0)
        mics = pra.MicrophoneArray(R, room.fs)
        room.add_microphone_array(mics)

        # Compute image sources
        room.image_source_model()

        # Simulate the propagation
        room.compute_rir()
        room.simulate(snr=SNR)

        print("MICROPHONES SIGNAL SHAPE FOR SOURCE", j + 1)
        print("Signal shape for every M:", room.mic_array.signals.shape)

        # Compute DOA Class
        doa_deg = sources_doa[j]
        doa_classes = []
        for n in range(37):
            doa_classes.append(0.0)
        doa_class = 0.0
        for d in range(len(DOAs) - 1):
            if DOAs[d] <= doa_deg < DOAs[d + 1]:
                doa_class = d
                doa_classes[d] = 1.0
        if doa_deg == 180:
            doa_class = 37
            doa_classes[36] = 1.0
        print("DOA:", doa_deg, ", DOA Class:", doa_class, ", DOA Classes:", doa_classes)

        # Compute phase of STFT
        sig = room.mic_array.signals
        sig_stft = tf.signal.stft(sig,
                                  window_fn=tf.signal.hann_window,
                                  frame_length=256,
                                  frame_step=128,
                                  fft_length=256
                                  )
        sig_stft_phase = tf.math.angle(sig_stft)
        sig_stft_phase = tf.reshape(sig_stft_phase,
                                    [sig_stft_phase.shape[0],
                                     sig_stft_phase.shape[2],
                                     sig_stft_phase.shape[1]]
                                    )
        print("Phase signals shape: ", sig_stft_phase.shape)

        if data_type == 'noise':
            slicing_indexes = np.arange(0, sig_stft_phase.shape[2] - 1, 1)
        else:
            slicing_indexes = random.sample(range(0, sig_stft_phase.shape[2] - 1), 102)

        for i in slicing_indexes:
            # Slicing
            phase_signal = tf.slice(
                sig_stft_phase,
                begin=[0, 0, i],
                size=[sig_stft_phase.shape[0], sig_stft_phase.shape[1], 1]
            )

            # Write
            feature0 = np.expand_dims(sources_pos[j], axis=0)
            feature1 = np.expand_dims(mic_center, axis=0)
            feature2 = np.expand_dims(room.mic_array.signals, axis=0)
            feature3 = np.expand_dims(phase_signal, axis=0)
            feature4 = np.expand_dims(phase_signal.shape, axis=0)
            feature5 = np.expand_dims(doa_deg, axis=0)
            feature6 = np.expand_dims(doa_classes, axis=0)
            dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3, feature4, feature5,
                                                          feature6))
            for feature0, feature1, feature2, feature3, feature4, feature5, feature6 in dataset:
                example = serialize_example(tf.io.serialize_tensor(feature0), tf.io.serialize_tensor(feature1),
                                            tf.io.serialize_tensor(feature2), tf.io.serialize_tensor(feature3),
                                            tf.io.serialize_tensor(feature4), tf.io.serialize_tensor(feature5),
                                            tf.io.serialize_tensor(feature6))
                writer.write(example)

        print("-------------------------------------------------------")


'''
# Old Version

# Specify reverberation and noise
T60 = 0.2
SNR = 20


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3, feature4, feature5, feature6):
    feature = {
        'source': _bytes_feature(feature0),
        'microphone': _bytes_feature(feature1),
        'signals': _bytes_feature(feature2),
        'phase_signals': _bytes_feature(feature3),
        'phase_signals_shape': _bytes_feature(feature4),
        'doa': _bytes_feature(feature5),
        'doa_classes': _bytes_feature(feature6),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Download dataset
corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl'])

# Specify room dimensions
corners = np.array([[0, 0], [0, 8.2], [3.6, 8.2], [3.6, 0]]).T  # [x,y]

# Sources generation [Room subdivision: 14*28*3 for 1176 elements, 15*30*5 for 2250 elements,
# 16*30*5 for 2400 elements, 19*33*8 for 5016 elements]
rows_number = 16
cols_number = 30
depths_number = 5

rows = np.arange(0, 3.6, 3.6 / rows_number)
cols = np.arange(0, 8.2, 8.2 / cols_number)
depths = np.arange(0.92, 1.53, (1.53 - 0.92) / depths_number)

noise_signals = []
sources_pos = []
for i in range(rows_number):
    for j in range(cols_number):
        for k in range(depths_number):
            pos = [rows[i], cols[j], depths[k]]
            sources_pos.append(pos)

            # 0 is the mean of the normal distribution you are choosing from
            # 1 is the standard deviation of the normal distribution
            # 10000 is the number of elements you get in array noise
            noise = np.random.normal(0, 1, 10000)
            noise_signals.append(noise)

# DOA classes
DOA_range = [0, 181]
resolution = 5
DOAs = np.arange(DOA_range[0], DOA_range[1], resolution)
print(DOAs)

# Number of examples: 1131
N = 2400

with tf.io.TFRecordWriter(filename) as writer:
    for j in range(0, N, 1):
        signal = noise_signals[j]
        fs = 16000

        # Add source to 3D room (set max_order to a low value for a quick, but less accurate, RIR)
        source_position = sources_pos[j]
        room = pra.Room.from_corners(corners, fs=fs, max_order=8, absorption=0.16 / T60)
        room.extrude(2.4)
        room.add_source(source_position, signal=signal)

        # Add microphones to 3D room
        mic_center = np.array([1.8, 4.5, 0.0])

        R = pra.linear_2D_array(mic_center[:2], M=4, phi=0, d=0.03)
        R = np.concatenate((R, np.ones((1, 4)) * mic_center[2]), axis=0)
        mics = pra.MicrophoneArray(R, room.fs)
        room.add_microphone_array(mics)

        # Compute image sources
        room.image_source_model()

        # Simulate the propagation
        room.compute_rir()
        room.simulate(snr=SNR)

        print("MICROPHONES SIGNAL SHAPE FOR SOURCE", j + 1)
        print("Signal shape for every M:", room.mic_array.signals.shape)

        # Compute DOA
        cat = sources_pos[j][2]
        ip = math.sqrt(math.pow((sources_pos[j][0] - mic_center[0]), 2) + math.pow((
                sources_pos[j][1] - mic_center[1]), 2) + math.pow((sources_pos[j][2] - mic_center[2]), 2))
        doa_rad = math.asin(cat / ip)
        doa_deg = (doa_rad * 180) / math.pi
        if sources_pos[j][0] > mic_center[0]:
            doa_deg = 180 - doa_deg

        # Compute DOA Class
        doa_classes = []
        for n in range(37):
            doa_classes.append(0.0)
        doa_class = 0.0
        for d in range(len(DOAs) - 1):
            if DOAs[d] <= doa_deg < DOAs[d + 1]:
                doa_class = d
                doa_classes[d] = 1.0
        if doa_deg == 180:
            doa_class = 37
            doa_classes[36] = 1.0
        print("DOA:", doa_deg, ", DOA Class:", doa_class, ", DOA Classes:", doa_classes)

        # Compute phase of STFT
        sig = room.mic_array.signals
        sig_stft = tf.signal.stft(sig, window_fn=tf.signal.hann_window, frame_length=256, frame_step=128,
                                  fft_length=256)
        sig_stft_phase = tf.math.angle(sig_stft)
        sig_stft_phase = tf.reshape(sig_stft_phase, [sig_stft_phase.shape[0], sig_stft_phase.shape[2],
                                                     sig_stft_phase.shape[1]])
        print("Phase signal shape: ", sig_stft_phase.shape)

        # Write
        feature0 = np.expand_dims(sources_pos[j], axis=0)
        feature1 = np.expand_dims(mic_center, axis=0)
        feature2 = np.expand_dims(room.mic_array.signals, axis=0)
        feature3 = np.expand_dims(sig_stft_phase, axis=0)
        feature4 = np.expand_dims(sig_stft_phase.shape, axis=0)
        feature5 = np.expand_dims(doa_deg, axis=0)
        feature6 = np.expand_dims(doa_classes, axis=0)
        dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3, feature4, feature5,
                                                      feature6))
        for feature0, feature1, feature2, feature3, feature4, feature5, feature6 in dataset:
            example = serialize_example(tf.io.serialize_tensor(feature0), tf.io.serialize_tensor(feature1),
                                        tf.io.serialize_tensor(feature2), tf.io.serialize_tensor(feature3),
                                        tf.io.serialize_tensor(feature4), tf.io.serialize_tensor(feature5),
                                        tf.io.serialize_tensor(feature6))
            writer.write(example)

            print("-------------------------------------------------------")
'''

'''
# Version 2

filename = '/nas/home/amontali/data_doa/doa2.tfrecord'


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(feature0,feature1,feature2,feature3,feature4,feature5,feature6):
    feature = {
       'source': _bytes_feature(feature0),
       'microphone': _bytes_feature(feature1),
       'signals': _bytes_feature(feature2),
       'phase_signals': _bytes_feature(feature3),
       'phase_signals_shape': _bytes_feature(feature4),
       'doa': _bytes_feature(feature5),
       'doa_classes': _bytes_feature(feature6),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Download dataset
corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl'])

# Specify rooms dimensions and reverberations
corners1 = np.array([[0, 0], [0, 6.], [6., 6.], [6., 0]]).T
corners2 = np.array([[0, 0], [0, 5.], [5., 5.], [5., 0]]).T
corners = [corners1, corners2]
T60 = np.array([0.3, 0.2])
alfa = 0.16 / T60

# Sources generation [Room subdivision: 14*28*3 for 1176 elements]
rows_number = 14
cols_number = 28
depths_number = 3

sources_pos = []
noise_signals = []
for r in range(2):
    rows = np.arange(0, corners[r][1][1], corners[r][1][1] / rows_number)
    cols = np.arange(0, corners[r][1][1], corners[r][1][1] / cols_number)
    depths = np.arange(0.92, 1.53, (1.53 - 0.92) / depths_number)

    noise_sigs = []
    s_pos = []
    for i in range(rows_number):
        for j in range(cols_number):
            for k in range(depths_number):
                pos = [rows[i], cols[j], depths[k]]
                s_pos.append(pos)

                # 0 is the mean of the normal distribution you are choosing from
                # 1 is the standard deviation of the normal distribution
                # 10000 is the number of elements you get in array noise
                noise = np.random.normal(0, 1, 20000)
                noise_sigs.append(noise)
    sources_pos.append(s_pos)
    noise_signals.append(noise_sigs)

# Microphones generation
mic_centers = []
for r in range(2):
    mic_centers_room = []
    for i in range(7):
        mic_pos = [np.random.uniform(0, corners[r][0][3]), np.random.uniform(0, corners[r][1][1]), 0.0]
        mic_centers_room.append(mic_pos)
    mic_centers.append(mic_centers_room)

# DOA classes
DOA_range = [0, 181]
resolution = 5
DOAs = np.arange(DOA_range[0], DOA_range[1], resolution)
print(DOAs)

N = len(corpus)

with tf.io.TFRecordWriter(filename) as writer:
    for j in range(0, N, 1):
        for r in range(2):
            signal = noise_signals[r][j]
            fs = 16000

            # Add source to 3D room
            source_position = sources_pos[r][j]
            room = pra.Room.from_corners(corners[r], fs=fs, max_order=8, absorption=alfa[r])  # set max_order to a low value for a quick (but less accurate) RIR
            room.extrude(2.5)
            room.add_source(source_position, signal=signal)

            for i in range(7):
                # Add microphones to 3D room
                R = pra.linear_2D_array(mic_centers[r][i][:2], M=4, phi=0, d=0.03)
                R = np.concatenate((R, np.ones((1, 4)) * mic_centers[r][i][2]), axis=0)
                mics = pra.MicrophoneArray(R, room.fs)
                room.add_microphone_array(mics)

                # compute image sources
                room.image_source_model()

                # Simulate the propagation
                room.compute_rir()
                room.simulate(snr=20)

                print("MICROPHONES SIGNAL SHAPE FOR ROOM", r + 1, "SOURCE", j + 1, "AND MICROPHONE", i + 1)
                print("Signal shape for each M:", room.mic_array.signals.shape)

                # Compute DOA
                cat = sources_pos[r][j][2]
                ip = math.sqrt((sources_pos[r][j][0] - mic_centers[r][i][0]) ** 2 + (
                            sources_pos[r][j][1] - mic_centers[r][i][1]) ** 2 + (
                                           sources_pos[r][j][2] - mic_centers[r][i][2]) ** 2)
                doa_rad = math.asin(cat / ip)
                doa_deg = (doa_rad * 180) / math.pi
                if sources_pos[r][j][0] > mic_centers[r][i][0]:
                    doa_deg = 180 - doa_deg

                # Compute DOA Class
                doa_classes = []
                for n in range(37):
                    doa_classes.append(0.0)
                doa_class = 0.0
                for d in range(len(DOAs) - 1):
                    if DOAs[d] <= doa_deg < DOAs[d + 1]:
                        doa_class = d + 1
                        doa_classes[d + 1] = 1.0
                if doa_deg == 180:
                    doa_class = 37
                    doa_classes[36] = 1.0
                print("DOA:", doa_deg, ", DOA Class:", doa_class, ", DOA Class:", doa_classes)

                # Compute phase of STFT
                sig = room.mic_array.signals
                sig_stft = tf.signal.stft(sig, window_fn=tf.signal.hann_window, frame_length=256, frame_step=128,
                                          fft_length=256)
                sig_stft_phase = tf.math.angle(sig_stft)
                sig_stft_phase = tf.reshape(sig_stft_phase,
                                            [sig_stft_phase.shape[0], sig_stft_phase.shape[2], sig_stft_phase.shape[1]])
                print("Phase signal shape: ", sig_stft_phase.shape)

                # Write
                feature0 = np.expand_dims(sources_pos[r][j], axis=0)
                feature1 = np.expand_dims(mic_centers[r][i], axis=0)
                feature2 = np.expand_dims(room.mic_array.signals, axis=0)
                feature3 = np.expand_dims(sig_stft_phase, axis=0)
                feature4 = np.expand_dims(sig_stft_phase.shape, axis=0)
                feature5 = np.expand_dims(doa_deg, axis=0)
                feature6 = np.expand_dims(doa_classes, axis=0)
                dataset = tf.data.Dataset.from_tensor_slices(
                    (feature0, feature1, feature2, feature3, feature4, feature5, feature6))
                for feature0, feature1, feature2, feature3, feature4, feature5, feature6 in dataset:
                    example = serialize_example(tf.io.serialize_tensor(feature0), tf.io.serialize_tensor(feature1),
                                                tf.io.serialize_tensor(feature2), tf.io.serialize_tensor(feature3),
                                                tf.io.serialize_tensor(feature4), tf.io.serialize_tensor(feature5),
                                                tf.io.serialize_tensor(feature6))
                    writer.write(example)

                print("-------------------------------------------------------")
'''
