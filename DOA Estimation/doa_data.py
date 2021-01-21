import numpy as np
import pyroomacoustics as pra
import tensorflow as tf
import math
import os
import argparse

parser = argparse.ArgumentParser(description='DOA data generation')
parser.add_argument('--filename', type=str, help='filename', default='/nas/home/amontali/data_doa')
parser.add_argument('--dataset_type', type=str, help='dataset_type', default='train')
parser.add_argument('--T60', type=float, help='T60', default=0.2)
parser.add_argument('--SNR', type=int, help='SNR', default=20)
parser.add_argument('--gpu', type=str, help='gpu', default='0')
args = parser.parse_args()
filename = args.filename
dataset_type = args.dataset_type
T60 = args.T60
SNR = args.SNR
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


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

# Specify microphone position
mic_center = np.array([1.8, 7.0, 0.92])

# DOA classes
DOA_range = [0, 181]
resolution = 5
DOAs = np.arange(DOA_range[0], DOA_range[1], resolution)

# Sources generation
if dataset_type == 'test':
    # 300 Test sources
    test_sources = []

    rows_number = 15
    cols_number = 20

    rows = np.arange(0.8, 2.8, (2.8 - 0.8) / rows_number)
    cols = np.arange(3.0, 5.0, (5.0 - 3.0) / cols_number)

    for i in range(rows_number):
        for j in range(cols_number):
            pos = [rows[i], cols[j], 0.92]
            test_sources.append(pos)

    print(len(test_sources))
    sources_pos = test_sources

else:
    # 1500 Train sources -> (1200 train + 300 val)
    sources_pos = []

    rows_number = 10
    cols_number = 50

    rows = np.arange(0.2, 0.8, (0.8 - 0.2) / rows_number)
    cols = np.arange(2.4, 5.6, (5.6 - 2.4) / cols_number)

    for i in range(rows_number):
        for j in range(cols_number):
            pos = [rows[i], cols[j], 0.92]
            sources_pos.append(pos)

    rows_number = 10
    cols_number = 50

    rows = np.arange(2.8, 3.4, (3.4 - 2.8) / cols_number)
    cols = np.arange(2.4, 5.6, (5.6 - 2.4) / cols_number)

    for i in range(rows_number):
        for j in range(cols_number):
            pos = [rows[i], cols[j], 0.92]
            sources_pos.append(pos)

    rows_number = 25
    cols_number = 10

    rows = np.arange(0.8, 2.8, (2.8 - 0.8) / rows_number)
    cols = np.arange(2.4, 3.0, (3.0 - 2.4) / cols_number)

    for i in range(rows_number):
        for j in range(cols_number):
            pos = [rows[i], cols[j], 0.92]
            sources_pos.append(pos)

    rows_number = 25
    cols_number = 10

    rows = np.arange(0.8, 2.8, (2.8 - 0.8) / rows_number)
    cols = np.arange(5.0, 5.6, (5.6 - 5.0) / cols_number)

    for i in range(rows_number):
        for j in range(cols_number):
            pos = [rows[i], cols[j], 0.92]
            sources_pos.append(pos)

    print(len(sources_pos))

    train_sources = []
    val_sources = []
    # val_indexes = random.sample(range(0, 1500), 300)
    val_indexes = [818, 1008, 1177, 391, 491, 1215, 186, 595, 1042, 257, 889, 49, 1433, 1451, 703, 518, 567, 596, 587, 1368, 790, 805, 1344, 486, 539, 147, 853, 197, 920, 1293, 1296, 281, 1149, 1202, 1478, 1184, 69, 121, 961, 1325, 440, 653, 728, 476, 181, 1030, 339, 1253, 1087, 1191, 1292, 771, 1006, 713, 1220, 1402, 77, 665, 1250, 731, 1233, 1086, 222, 770, 871, 1411, 1431, 885, 558, 599, 1328, 1365, 363, 254, 1065, 948, 50, 1097, 856, 1379, 1070, 311, 463, 1476, 573, 1137, 1255, 743, 473, 124, 1485, 717, 605, 723, 426, 395, 990, 1132, 1166, 569, 808, 326, 221, 1011, 838, 79, 1285, 343, 973, 175, 227, 1270, 483, 1120, 659, 1284, 1156, 1021, 1499, 874, 1450, 909, 937, 1155, 575, 1231, 1228, 226, 410, 40, 199, 1083, 368, 494, 385, 416, 848, 1276, 814, 1390, 1299, 1452, 726, 429, 75, 200, 1466, 456, 437, 900, 89, 258, 958, 236, 694, 750, 1106, 1201, 608, 361, 1354, 1377, 986, 1259, 288, 908, 403, 578, 62, 704, 230, 637, 816, 563, 66, 418, 647, 583, 1405, 350, 1470, 1426, 956, 41, 516, 384, 530, 1415, 1265, 555, 319, 936, 584, 593, 1317, 253, 1152, 22, 390, 1391, 392, 1186, 591, 1403, 627, 689, 612, 154, 917, 725, 729, 1052, 169, 602, 538, 1300, 671, 355, 999, 414, 926, 1111, 933, 1015, 1236, 537, 1495, 1381, 654, 666, 691, 1316, 459, 204, 155, 223, 78, 228, 19, 1055, 616, 177, 686, 4, 316, 552, 1135, 913, 1129, 758, 397, 252, 399, 777, 1258, 504, 991, 1318, 987, 617, 398, 1179, 542, 276, 1465, 417, 923, 813, 905, 774, 527, 872, 362, 613, 156, 802, 784, 1219, 128, 682, 388, 1237, 1098, 441, 80, 570, 1246, 680, 232, 1498, 1398, 793, 209, 1462, 379, 1093, 337, 619, 1267, 610]
    for i in range(len(sources_pos)):
        for j in val_indexes:
            if i == j:
                val_sources.append(sources_pos[i])

    for s in sources_pos:
        if s not in val_sources:
            train_sources.append(s)

    print(len(train_sources))
    print(len(val_sources))

    if dataset_type == 'train':
        sources_pos = train_sources
    else:
        sources_pos = val_sources

# Compute DOA
sources_doa = []
for source in sources_pos:
    dist = math.sqrt(math.pow(source[0] - mic_center[0], 2) + math.pow(source[1] - mic_center[1], 2) +
                     math.pow(source[2] - mic_center[2], 2))
    if source[0] < mic_center[0]:
        teta = (math.acos((mic_center[0] - source[0]) / dist) * 180) / math.pi
    else:
        teta = 180 - (math.acos((source[0] - mic_center[0]) / dist) * 180) / math.pi
    sources_doa.append(teta)

# Number of examples
N = len(sources_pos)

# Download speech dataset
corpus = pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl'])

with tf.io.TFRecordWriter(filename) as writer:
    for j in range(0, N, 1):
        if j < len(corpus):
            signal = corpus[j].data
            fs = corpus[j].fs
        else:
            signal = corpus[j - len(corpus)].data
            fs = corpus[j - len(corpus)].fs

        # Add source to 3D room
        source_position = sources_pos[j]
        room = pra.Room.from_corners(corners, fs=fs, max_order=8, absorption=0.16 / T60)
        room.extrude(2.4)
        room.add_source(source_position, signal=signal)

        # Add microphones to 3D room
        R = pra.circular_2D_array(mic_center[:2], M=8, phi0=0, radius=0.1)
        R = np.concatenate((R, np.ones((1, 8)) * mic_center[2]), axis=0)
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

