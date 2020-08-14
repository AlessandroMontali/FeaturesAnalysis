import tensorflow as tf


# Create dataset for endtoend


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


def create_endtoend_dataset(filepath, buffer_size, batch_size, train_size, dataset_type):
    """
        Create train and test datasets for endtoend source localization architecture
            Args:
                filepath: String. Path to TFRecords Folder containing the dataset
                buffer_size: Integer. Number of element from this dataset from which the new dataset will sample
                batch_size: Integer. Number of consecutive elements of this dataset to combine in a single batch
                dataset_type: String. Dataset returned (train, test or entire dataset)
            Returns:
                dataset (train, test or entire dataset)
    """

    def parse_function(serialized_example):
        feature_description = {
            'source': tf.io.FixedLenFeature((), tf.string),
            'microphone': tf.io.FixedLenFeature((), tf.string),
            'signals': tf.io.FixedLenFeature((), tf.string),
            'win_signals_1': tf.io.FixedLenFeature((), tf.string),
            'win_signals_2': tf.io.FixedLenFeature((), tf.string),
            'win_signals_3': tf.io.FixedLenFeature((), tf.string),
            'win_signals_1_shape': tf.io.FixedLenFeature((), tf.string),
            'win_signals_2_shape': tf.io.FixedLenFeature((), tf.string),
            'win_signals_3_shape': tf.io.FixedLenFeature((), tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature_description)
        sources = tf.io.parse_tensor(example['source'], out_type=tf.float64)
        win_signals = tf.io.parse_tensor(example['win_signals_1'], out_type=tf.float64)
        win_signals_shape = tf.io.parse_tensor(example['win_signals_1_shape'], out_type=tf.int64)

        # Randomly slice one dimension relative to one window of the input signal
        slice_idx = tf.random.uniform(
            shape=(), minval=0,
            maxval=win_signals_shape[2] - 1,
            dtype=tf.dtypes.int64
        )

        # Slicing
        win_signals = tf.slice(
            win_signals,
            begin=[0, 0, slice_idx],
            size=[win_signals_shape[0], win_signals_shape[1], 1]
        )

        # Squeeze third dimension
        win_signals = tf.squeeze(
            win_signals,
            axis=2
        )

        # Reshape windowed signals
        win_signals = tf.reshape(win_signals, [win_signals_shape[1], win_signals_shape[0]])

        # Normalize signals between 0 and 1
        win_signals = normalize(win_signals)

        # Specify shape to remove fit error
        win_signals = tf.reshape(win_signals, shape=(1280, 8))
        sources = tf.reshape(sources, shape=(3,))

        return win_signals, sources

    tfrecord_dataset = tf.data.TFRecordDataset(filepath)
    dataset = tfrecord_dataset.map(parse_function)

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size)

    # Set the batch size and return train or test dataset
    if dataset_type == 'train':
        return dataset.take(train_size).batch(batch_size).repeat()
    if dataset_type == 'test':
        return dataset.skip(train_size).batch(batch_size).repeat()
    else:
        return dataset


