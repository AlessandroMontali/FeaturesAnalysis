import tensorflow as tf


# Create dataset for doa


def normalize(input_tensor):
    """
        Normalize tensors between 0 and 1

        Args:
            input_tensor
        Returns:
            normalized tensor
    """

    max_val = tf.math.reduce_max(input_tensor)
    min_val = tf.math.reduce_min(input_tensor)

    normalized_tensor = tf.math.divide(
      input_tensor - min_val,
      max_val - min_val
    )

    return normalized_tensor


def create_doa_dataset(filepath, buffer_size):
    """
        Create train and test datasets for doa estimation architecture
            Args:
                filepath: String. Path to TFRecords Folder containing the dataset
                buffer_size: Integer. Number of element from this dataset from which the new dataset will sample
            Returns:
                dataset
    """

    def parse_function(serialized_example):
        feature_description = {
            'source': tf.io.FixedLenFeature((), tf.string),
            'microphone': tf.io.FixedLenFeature((), tf.string),
            'signals': tf.io.FixedLenFeature((), tf.string),
            'phase_signals': tf.io.FixedLenFeature((), tf.string),
            'phase_signals_shape': tf.io.FixedLenFeature((), tf.string),
            'doa': tf.io.FixedLenFeature((), tf.string),
            'doa_classes': tf.io.FixedLenFeature((), tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        doa_classes = tf.io.parse_tensor(example['doa_classes'], out_type=tf.float64)
        phase_signals = tf.io.parse_tensor(example['phase_signals'], out_type=tf.float64)
        phase_signals_shape = tf.io.parse_tensor(example['phase_signals_shape'], out_type=tf.int64)

        # Randomly slice one dimension relative to one window of the input signal
        slice_idx = tf.random.uniform(
            shape=(), minval=0,
            maxval=phase_signals_shape[2] - 1,
            dtype=tf.dtypes.int64
        )

        # Slicing
        phase_signals = tf.slice(
            phase_signals,
            begin=[0, 0, slice_idx],
            size=[phase_signals_shape[0], phase_signals_shape[1], 1]
        )

        # Normalize signals between 0 and 1
        phase_signals = normalize(phase_signals)

        # Specify shape to remove fit error
        phase_signals = tf.reshape(phase_signals, shape=(8, 129, 1))
        doa_classes = tf.reshape(doa_classes, shape=(37,))

        return phase_signals, doa_classes

    tfrecord_dataset = tf.data.TFRecordDataset(filepath)
    dataset = tfrecord_dataset.map(parse_function)

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size)

    return dataset

