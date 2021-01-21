import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import mean_absolute_error
from innvestigate.analyzer.base import ReverseAnalyzerBase
import innvestigate.utils.keras.graph as kgraph
import innvestigate.utils.keras.checks as kchecks
import innvestigate.utils.keras as kutils
import innvestigate.utils as iutils
import innvestigate.layers as ilayers
from doa_data_provider import create_doa_dataset
import os

# Variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
filename = '/nas/home/amontali/data_doa/speech_T60_1.2_SNR_10_test.tfrecord'
checkpoint_filepath = '/nas/home/amontali/models_doa/speech_T60_1.2_SNR_10'
save_filepath = '/nas/home/amontali/presentation_plots/doa_T60_1.2_SNR_10.pdf'

# Load dataset
test_dataset = create_doa_dataset(filepath=filename, buffer_size=300)

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


def average(lst):
    return sum(lst) / len(lst)


# Compute MAE
DOAs = np.arange(0, 181, 5)
deg_errors = []
for data in test_dataset:
    prediction = model.predict(tf.expand_dims(data[0], axis=0))
    deg_errors.append(mean_absolute_error([DOAs[np.argmax(data[1])]], [DOAs[np.argmax(prediction[0])]]))
mae = average(deg_errors)
print("MAE:", mae)


# Plot input signals and relevance signals
analyzer = SimpleLRP(model)
for data in test_dataset.take(1):
    print('DOA: ', data[1])
    input_signal = tf.expand_dims(data[0], axis=0)   # (1, 8, 129, 1)
    analysis = analyzer.analyze(input_signal)        # (1, 8, 129, 1)
    result_lrp = tf.squeeze(analysis)                # (8, 129)
    signal = tf.squeeze(data[0])                     # (8, 129)

    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(np.unwrap(2 * signal[i+4]) / 2)
        if i == 0:
            ax.set_ylabel(r'$\phi_5(k)$', fontsize=20)
        if i == 1:
            ax.set_ylabel(r'$\phi_6(k)$', fontsize=20)
        ax.tick_params(labelsize=15)
    plt.xlabel('k', fontsize=20)
    plt.tight_layout()
    plt.savefig('/nas/home/amontali/presentation_plots/doa_inputSig.pdf')
    plt.show()

    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(np.unwrap(2 * result_lrp[i+4]) / 2, color='r')
        if i == 0:
            ax.set_ylabel(r'$R_5(k)$', fontsize=20)
        if i == 1:
            ax.set_ylabel(r'$R_6(k)$', fontsize=20)
        ax.tick_params(labelsize=15)
    plt.xlabel('k', fontsize=20)
    plt.tight_layout()
    plt.savefig('/nas/home/amontali/presentation_plots/doa_RelevanceSig.pdf')
    plt.show()


# Manipulate a fraction of the input signal by setting selected samples to zero:
# - samples are selected at random.
# - samples are selected according to maximal relevance as attributed by LRP.

analyzer = SimpleLRP(model)


def analysis_zero_samples(test_data, percentages, analysis_type):
    acc_values = []
    for perc in percentages:
        percentage = int(129 * perc / 100)

        acc_zero_samples = []
        for data in test_data:
            signal = data[0]  # (8, 129, 1)
            doa = data[1]

            indices_max_values = []
            if analysis_type == 'lrp':
                input_signal = tf.expand_dims(signal, axis=0)  # (1, 8, 129, 1)
                analysis = analyzer.analyze(input_signal)      # (1, 8, 129, 1)
                result_lrp = tf.squeeze(analysis)              # (8, 129)

                for i in range(4):
                    indices = (-result_lrp[i].numpy()).argsort()[:percentage]
                    indices_max_values.append(indices)

            if analysis_type == 'random':
                for i in range(4):
                    indices_max_values.append(random.sample(range(0, 128), percentage))

            input_sig = tf.squeeze(signal)  # (8, 129)
            signals_zero_samples = input_sig.numpy()

            for i in range(4):
                for j in range(129):
                    for k in range(len(indices_max_values[i])):
                        if j == indices_max_values[i][k]:
                            signals_zero_samples[i][j] = 0.0

            signals_zero_samples = tf.convert_to_tensor(signals_zero_samples)
            
            signals_zero_samples = tf.expand_dims(signals_zero_samples, axis=0)   # (1, 8, 129)
            signals_zero_samples = tf.expand_dims(signals_zero_samples, axis=-1)  # (1, 8, 129, 1)
            prediction = model.predict(signals_zero_samples)
            acc_zero_samples.append(mean_absolute_error([DOAs[np.argmax(doa)]], [DOAs[np.argmax(prediction[0])]]))

        acc_values.append(average(acc_zero_samples))
        print(acc_values)

    print(acc_values)
    return acc_values


percentages_zero_samples = [0, 10, 20, 30, 50, 70]

plt.figure()
plt.plot(percentages_zero_samples, analysis_zero_samples(test_dataset, percentages_zero_samples, 'lrp'),
         label="LRP")
plt.plot(percentages_zero_samples, analysis_zero_samples(test_dataset, percentages_zero_samples, 'random'),
         label="Random")
plt.xlabel('Signals samples set to zero [%]', fontsize=20)
plt.ylabel('MAE [°]', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(prop={'size': 20})
plt.tight_layout()
plt.savefig(save_filepath)
plt.show()

