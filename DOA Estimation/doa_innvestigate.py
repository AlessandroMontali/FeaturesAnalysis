import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import accuracy_score
from innvestigate.analyzer.base import ReverseAnalyzerBase
import innvestigate.utils.keras.graph as kgraph
import innvestigate.utils.keras.checks as kchecks
import innvestigate.utils.keras as kutils
import innvestigate.utils as iutils
import innvestigate.layers as ilayers
from doa_data_provider import create_doa_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Load dataset
filename = '/nas/home/amontali/data_doa/doa.tfrecord'
dataset = create_doa_dataset(filepath=filename, buffer_size=24480)
test_dataset = dataset.take(5000).batch(100).repeat()

# Load model
checkpoint_filepath = '/nas/home/amontali/models_doa/doa'
model = tf.keras.models.load_model(checkpoint_filepath)

# Evaluate model
model.evaluate(test_dataset, steps=50)


class SimpleLRPRule(kgraph.ReverseMappingBase):

    # Basic LRP decomposition rule.

    def __init__(self, layer, state, bias=True):
        # Copy forward layer
        self._layer_wo_act = layer

    def apply(self, Xs, Ys, Rs, reverse_state):

        grad = ilayers.GradientWRT(len(Xs))

        # Get layers
        Zs = kutils.apply(self._layer_wo_act, Xs)

        # Divide incoming relevance by the layers
        tmp = [ilayers.SafeDivide()([a, b]) for a, b in zip(Rs, Zs)]

        # Propagate the relevance to input neurons using the gradient
        tmp = iutils.to_list(grad(Xs + Zs + tmp))

        # Re-weight relevance with the input values
        return [tf.keras.layers.Multiply()([a, b]) for a, b in zip(Xs, tmp)]


class SimpleLRP(ReverseAnalyzerBase):
    def _default_reverse_mapping(self, Xs, Ys, reversed_Ys, reverse_state):
        return ilayers.GradientWRT(len(Xs))(Xs + Ys + reversed_Ys)

    def _head_mapping(self, X):
        # Keeping the output signal
        return X

    def _create_analysis(self, *args, **kwargs):
        self._add_conditional_reverse_mapping(
            # Apply to all layers that contain a kernel
            lambda layer: kchecks.contains_kernel(layer),
            SimpleLRPRule,
            name="z_rule",
        )

        return super(SimpleLRP, self)._create_analysis(*args, **kwargs)


# Manipulate a fraction of the input signal by setting selected samples to zero:
# - samples are selected at random.
# - samples are selected according to maximal relevance as attributed by LRP.

analyzer = SimpleLRP(model)


def find_max_indexes(a, N):
    b = a[:]
    locations = []
    minimum = min(b) - 1
    for i in range(N):
        maxIndex = b.index(max(b))
        locations.append(maxIndex)
        b[maxIndex] = minimum

    return locations


def average(lst):
    return sum(lst) / len(lst)


def analysis_zero_samples(test_data, percentages, analysis_type):
    acc_values = []
    for perc in percentages:
        percentage = int(129 * perc / 100)

        acc_zero_samples = []
        for data in test_data:
            signal = data[0]  # (4, 129, 1)
            doa = data[1]

            indices_max_values = []
            if analysis_type == 'lrp':
                input_signal = tf.expand_dims(signal, axis=0)  # (1, 4, 129, 1)
                analysis = analyzer.analyze(input_signal)      # (1, 4, 129, 1)
                result_lrp = tf.squeeze(analysis)              # (4, 129)

                for i in range(4):
                    indices_max_values.append(find_max_indexes(result_lrp[i].numpy().tolist(), percentage))

            if analysis_type == 'random':
                for i in range(4):
                    indices_max_values.append(random.sample(range(0, 128), percentage))

            input_sig = tf.squeeze(signal)  # (4, 129)
            signals_zero_samples = input_sig.numpy().tolist()

            for i in range(4):
                for j in range(129):
                    for k in range(len(indices_max_values[i])):
                        if j == indices_max_values[i][k]:
                            signals_zero_samples[i][j] = 0.0

            signals_zero_samples = tf.convert_to_tensor(signals_zero_samples)

            signals_zero_samples = tf.expand_dims(signals_zero_samples, axis=0)   # (1, 4, 129)
            signals_zero_samples = tf.expand_dims(signals_zero_samples, axis=-1)  # (1, 4, 129, 1)
            prediction = model.predict(signals_zero_samples)
            doa = tf.reshape(doa, (37, 1))
            prediction = prediction[0].reshape((37, 1))
            acc = accuracy_score(doa.numpy().tolist(), prediction.round())
            acc_zero_samples.append(acc)

        acc_values.append(average(acc_zero_samples))
        print(acc_values)

    print(acc_values)
    return acc_values


percentages_zero_samples = [0, 10, 20, 30, 50, 70]
dataset = create_doa_dataset(filepath=filename, buffer_size=24480)
test_dataset = dataset.take(5000)

plt.figure()
plt.plot(percentages_zero_samples, analysis_zero_samples(test_dataset, percentages_zero_samples, 'lrp'),
         label="LRP")
plt.plot(percentages_zero_samples, analysis_zero_samples(test_dataset, percentages_zero_samples, 'random'),
         label="Random")
plt.xlabel('% signal samples set to zero')
plt.ylabel('Accuracy')
plt.legend()
plt.title('DOA Analysis')
plt.show()
