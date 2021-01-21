import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import mean_squared_error
from innvestigate.analyzer.base import ReverseAnalyzerBase
import innvestigate.utils.keras.graph as kgraph
import innvestigate.utils.keras.checks as kchecks
import innvestigate.utils.keras as kutils
import innvestigate.utils as iutils
import innvestigate.layers as ilayers
from endtoend_data_provider import create_endtoend_dataset
import os

# Variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
filename = '/nas/home/amontali/data_endtoend/raw_audio/exp1_T60_1.2_SNR_-5_test.tfrecord'
checkpoint_filepath = '/nas/home/amontali/models_endtoend/raw_audio/exp1_T60_1.2_SNR_-5'
save_filepath = '/nas/home/amontali/presentation_plots/setup2_T60_1.2_SNR_-5.pdf'

# Load dataset
test_dataset = create_endtoend_dataset(filepath=filename, buffer_size=300)
eval_dataset = test_dataset.repeat().batch(100)

# Load model
model = tf.keras.models.load_model(checkpoint_filepath)


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

        # Propagate the relevance to input neurons
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
    rmse_values = []
    for perc in percentages:
        percentage = int(1280 * perc / 100)

        rmse_zero_samples = []
        for data in test_data:
            signal = data[0]  # (1280, 8)
            source = data[1]

            indices_max_values = []
            if analysis_type == 'lrp':
                input_signal = tf.expand_dims(signal, axis=0)    # (1, 1280, 8)
                analysis = analyzer.analyze(input_signal)        # (1, 1280, 8)
                result_lrp = tf.transpose(tf.squeeze(analysis))  # (8, 1280)

                for i in range(8):
                    indices_max_values.append(find_max_indexes(result_lrp[i].numpy().tolist(), percentage))

            if analysis_type == 'random':
                for i in range(8):
                    indices_max_values.append(random.sample(range(0, 1279), percentage))

            input_sig = tf.transpose(signal)  # (8, 1280)
            signals_zero_samples = input_sig.numpy().tolist()

            for i in range(8):
                for j in range(1280):
                    for k in range(len(indices_max_values[i])):
                        if j == indices_max_values[i][k]:
                            signals_zero_samples[i][j] = 0.0

            signals_zero_samples = tf.convert_to_tensor(signals_zero_samples)

            signals_zero_samples = tf.expand_dims(tf.transpose(signals_zero_samples), axis=0)  # (1, 1280, 8)
            prediction = model.predict(signals_zero_samples)

            mse = mean_squared_error(source.numpy().tolist(), prediction[0])
            rmse_zero_samples.append(math.sqrt(mse))

        rmse_values.append(average(rmse_zero_samples))
        print(rmse_values)

    print(rmse_values)
    return rmse_values


percentages_zero_samples = [0, 10, 20, 30, 50, 70]

plt.figure()
plt.plot(percentages_zero_samples, analysis_zero_samples(test_dataset, percentages_zero_samples, 'lrp'),
         label="LRP")
plt.plot(percentages_zero_samples, analysis_zero_samples(test_dataset, percentages_zero_samples, 'random'),
         label="Random")
plt.xlabel('Signals samples set to zero [%]', fontsize=20)
plt.ylabel('RMSE [m]', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop={'size': 20})
plt.tight_layout()
plt.savefig(save_filepath)
plt.show()
