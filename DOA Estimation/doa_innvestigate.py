import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
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

data_type = 'speech'

# Load dataset
if data_type == 'noise':
    filename = '/nas/home/amontali/data_doa/doa.tfrecord'
    buffer_size = 24480
    test_size = 5000
    steps = 50
else:
    filename = '/nas/home/amontali/data_doa/doa_speech.tfrecord'
    buffer_size = 12240
    test_size = 2500
    steps = 25
batch_size = 100

dataset = create_doa_dataset(filepath=filename, buffer_size=buffer_size)
test_dataset = dataset.take(test_size).batch(batch_size).repeat()

# Load model
checkpoint_filepath = '/nas/home/amontali/models_doa/doa'
model = tf.keras.models.load_model(checkpoint_filepath)

# Evaluate model
model.evaluate(test_dataset, steps=steps)


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
                    indices = (-result_lrp[i].numpy()).argsort()[:percentage]
                    indices_max_values.append(indices)

            if analysis_type == 'random':
                for i in range(4):
                    indices_max_values.append(random.sample(range(0, 128), percentage))

            input_sig = tf.squeeze(signal)  # (4, 129)
            signals_zero_samples = input_sig.numpy()

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
            acc = accuracy_score(doa.numpy(), prediction.round())
            acc_zero_samples.append(acc)

        acc_values.append(average(acc_zero_samples))
        print(acc_values)

    print(acc_values)
    return acc_values


percentages_zero_samples = [0, 10, 20, 30, 50, 70]
dataset = create_doa_dataset(filepath=filename, buffer_size=buffer_size)
test_dataset = dataset.take(test_size)

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

'''
# Example sources for analysis

signals_for_analysis = []
sources_for_analysis = []
for data in dataset.take(3):
    signals_for_analysis.append(data[0])
    sources_for_analysis.append(data[1])
        
        
# Plot input signal with colors based on analysis

def mapping_signal_analysis(value, avg, max, min):
    value_norm = (value - min) / (max - min)
    avg_norm = (avg - min) / (max - min)

    if avg_norm > 0.5:
        if 0 <= value_norm < avg_norm / 3:
            return 1
        if avg_norm / 3 <= value_norm < avg_norm - avg_norm * 0.1:
            return 2
        if avg_norm - avg_norm * 0.1 <= value_norm < avg_norm:
            return 3
        if avg_norm <= value_norm < avg_norm + avg_norm * 0.1:
            return 4
        if avg_norm + avg_norm * 0.1 <= value_norm < avg_norm + avg_norm / 3:
            return 5
        if avg_norm + avg_norm / 3 <= value_norm < 1:
            return 6
    else:
        if 0 <= value_norm < avg_norm / 2:
            return 1
        if avg_norm / 2 <= value_norm < avg_norm - avg_norm * 0.1:
            return 2
        if avg_norm - avg_norm * 0.1 <= value_norm < avg_norm:
            return 3
        if avg_norm <= value_norm < avg_norm + avg_norm * 0.1:
            return 4
        if avg_norm + avg_norm * 0.1 <= value_norm < avg_norm + avg_norm / 2:
            return 5
        if avg_norm + avg_norm / 2 <= value_norm < 1:
            return 6


pp = PdfPages('/nas/home/amontali/tmp/doa_plots.pdf')

for s in range(len(signals_for_analysis)):
    # Source considered for analysis
    print(sources_for_analysis[s])
    source = str(np.round(sources_for_analysis[s].numpy(), 2).tolist())[1:-1]

    # Signal considered for analysis [dimension for visualization: (4, 129)]
    print(signals_for_analysis[s])
    input_sig = tf.squeeze(signals_for_analysis[s])

    # Creating the analyzer
    analyzer = SimpleLRP(model)

    # Input signal dimension for analysis: (1, 4, 129, 1)
    input_signal = tf.expand_dims(signals_for_analysis[s], axis=0)

    # Applying the analyzer
    analysis = analyzer.analyze(input_signal)

    # Result dimension for visualization: (4, 129)
    result_lrp = tf.squeeze(analysis)

    # Take max and min value for mapping
    max_val = tf.reduce_max(result_lrp)
    min_val = tf.reduce_min(result_lrp)
    avg_heatmap_signal = tf.reduce_mean(result_lrp)

    # Plot input signal colored with heatmap values from analysis
    fig, axes = plt.subplots(ncols=1, nrows=4, figsize=(5, 3))
    for i, ax in enumerate(axes.flatten()):
        one_values = np.copy(input_sig[i].numpy())
        two_values = np.copy(input_sig[i].numpy())
        three_values = np.copy(input_sig[i].numpy())
        four_values = np.copy(input_sig[i].numpy())
        five_values = np.copy(input_sig[i].numpy())
        six_values = np.copy(input_sig[i].numpy())
        heatmap_signal = np.copy(result_lrp[i].numpy())

        for elem in range(len(heatmap_signal)):
            if mapping_signal_analysis(heatmap_signal[elem], avg_heatmap_signal, max_val, min_val) == 1:
                five_values[elem] = np.nan
                four_values[elem] = np.nan
                three_values[elem] = np.nan
                two_values[elem] = np.nan
                one_values[elem] = np.nan
            if mapping_signal_analysis(heatmap_signal[elem], avg_heatmap_signal, max_val, min_val) == 2:
                six_values[elem] = np.nan
                four_values[elem] = np.nan
                three_values[elem] = np.nan
                two_values[elem] = np.nan
                one_values[elem] = np.nan
            if mapping_signal_analysis(heatmap_signal[elem], avg_heatmap_signal, max_val, min_val) == 3:
                six_values[elem] = np.nan
                five_values[elem] = np.nan
                three_values[elem] = np.nan
                two_values[elem] = np.nan
                one_values[elem] = np.nan
            if mapping_signal_analysis(heatmap_signal[elem], avg_heatmap_signal, max_val, min_val) == 4:
                six_values[elem] = np.nan
                five_values[elem] = np.nan
                four_values[elem] = np.nan
                two_values[elem] = np.nan
                one_values[elem] = np.nan
            if mapping_signal_analysis(heatmap_signal[elem], avg_heatmap_signal, max_val, min_val) == 5:
                six_values[elem] = np.nan
                five_values[elem] = np.nan
                four_values[elem] = np.nan
                three_values[elem] = np.nan
                one_values[elem] = np.nan
            if mapping_signal_analysis(heatmap_signal[elem], avg_heatmap_signal, max_val, min_val) == 6:
                six_values[elem] = np.nan
                five_values[elem] = np.nan
                four_values[elem] = np.nan
                three_values[elem] = np.nan
                two_values[elem] = np.nan

        # print(one_values)
        # print(two_values)
        # print(three_values)
        # print(four_values)
        # print(five_values)
        # print(six_values)
        ax.plot(one_values, color='maroon')
        ax.plot(two_values, color='red')
        ax.plot(three_values, color='orange')
        ax.plot(four_values, color='cyan')
        ax.plot(five_values, color='blue')
        ax.plot(six_values, color='darkblue')
    # plt.suptitle(source)
    pp.savefig()
    plt.show()

pp.close()
'''
