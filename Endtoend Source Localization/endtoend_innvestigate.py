import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import innvestigate
import seaborn as sb
from random import randrange
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Load dataset
filename = '/nas/home/amontali/data_endtoend/T60_0.5_SNR_30.tfrecord'
dataset = create_endtoend_dataset(filepath=filename, buffer_size=2262, batch_size='', train_size='', dataset_type='all')

# Load model
checkpoint_filepath = '/nas/home/amontali/models_endtoend/T60_0.5_SNR_10'
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


test_dataset = dataset.take(500)
percentages_zero_samples = [0, 5, 10, 20, 30, 50, 70]

plt.figure()
plt.plot(percentages_zero_samples, analysis_zero_samples(test_dataset, percentages_zero_samples, 'lrp'),
         label="LRP")
plt.plot(percentages_zero_samples, analysis_zero_samples(test_dataset, percentages_zero_samples, 'random'),
         label="Random")
plt.xlabel('% signal samples set to zero')
plt.ylabel('RMSE')
plt.legend()
plt.title('TRAIN: T60 = 0.5 s | SNR = 10 dB, TEST: T60 = 0.5 s | SNR = 30 dB')
plt.show()


'''
# Example sources for analysis

signals = []
sources = []
for data in dataset:
    signals.append(data[0])
    sources.append(data[1])

source_1 = np.asarray([0.77, 2.93, 0.92], np.float64)
source_2 = np.asarray([0.00, 2.93, 0.92], np.float64)
source_3 = np.asarray([0.77, 0.88, 0.92], np.float64)
source_4 = np.asarray([0.00, 0.88, 0.92], np.float64)

signals_for_analysis = []
sources_for_analysis = []
for s in range(len(sources)):
    source_compare = np.round(sources[s].numpy(), 2)
    if np.array_equal(source_compare, source_1) or np.array_equal(source_compare, source_2) or np.array_equal(
            source_compare, source_3) or np.array_equal(source_compare, source_4):
        signals_for_analysis.append(signals[s])
        sources_for_analysis.append(sources[s])
'''


'''
# Manipulate a fraction of the input signal by setting selected samples to zero (single signal)

source = str(np.round(sources_for_analysis[0].numpy(), 2).tolist())[1:-1]
print("Source:", source)
prediction = model.predict(tf.expand_dims(signals_for_analysis[0], axis=0))
print("Prediction:", prediction[0])

actual = sources_for_analysis[0].numpy().tolist()
predicted = prediction[0]
mse = mean_squared_error(actual, predicted)
rmse = math.sqrt(mse)
print("RMSE:", rmse)

analyzer = SimpleLRP(model)

input_signal = tf.expand_dims(signals_for_analysis[0], axis=0)
analysis = analyzer.analyze(input_signal)
result_lrp = tf.transpose(tf.squeeze(analysis))

percentage_zero_samples = 10
percentage_zero_samples = int(1280*percentage_zero_samples/100)
indices_max_values = []
for i in range(8):
    # indices_max_values.append(np.argpartition(result_lrp[i].numpy(), -percentage_zero_samples)[-percentage_zero_samples:])
    indices_max_values.append(find_max_indexes(result_lrp[i].numpy().tolist(), percentage_zero_samples))
print(indices_max_values) 

input_sig = tf.transpose(signals_for_analysis[0])

signals_zero_samples = input_sig.numpy().tolist()
for i in range(8):
    for j in range(1280):
        for k in range(len(indices_max_values[i])):
            if j == indices_max_values[i][k]:
                signals_zero_samples[i][j] = 0.0

signals_zero_samples = tf.convert_to_tensor(signals_zero_samples)

signals_zero_samples = tf.expand_dims(tf.transpose(signals_zero_samples), axis=0)
print("Source:", source)
prediction = model.predict(signals_zero_samples)
print("Prediction:", prediction[0])

actual = sources_for_analysis[0].numpy().tolist()
predicted = prediction[0]
mse = mean_squared_error(actual, predicted)
rmse = math.sqrt(mse)
print("RMSE:", rmse)

percentage_zero_samples = 10
percentage_zero_samples = int(1280*percentage_zero_samples/100)
indices_max_values_random = []
for i in range(8):
    indices_max_values_random.append(random.sample(range(0, 1280), percentage_zero_samples))
print(indices_max_values_random)

signals_zero_samples_random = input_sig.numpy().tolist()
for i in range(8):
    for j in range(1280):
        for k in range(len(indices_max_values_random[i])):
            if j == indices_max_values_random[i][k]:
                signals_zero_samples_random[i][j] = 0.0

signals_zero_samples_random = tf.convert_to_tensor(signals_zero_samples_random)

signals_zero_samples_random = tf.expand_dims(tf.transpose(signals_zero_samples_random), axis=0)
print("Source:", source)
prediction = model.predict(signals_zero_samples_random)
print("Prediction:", prediction[0])

actual = sources_for_analysis[0].numpy().tolist()
predicted = prediction[0]
mse = mean_squared_error(actual, predicted)
rmse = math.sqrt(mse)
print("RMSE:", rmse)
'''

'''
# Plot input signal with colors based on analysis (binary)

source = str(np.round(sources_for_analysis[0].numpy(), 2).tolist())[1:-1]
input_sig = tf.transpose(signals_for_analysis[0])

analyzer = SimpleLRP(model)

input_signal = tf.expand_dims(signals_for_analysis[0], axis=0)
analysis = analyzer.analyze(input_signal)
result_lrp = tf.transpose(tf.squeeze(analysis))


def average(lst):
    return sum(lst) / len(lst)


fig, axes = plt.subplots(ncols=1, nrows=8, figsize=(12, 6))
for i, ax in enumerate(axes.flatten()):
    pos_signal = tf.identity(input_sig[i]).numpy().tolist()
    neg_signal = tf.identity(input_sig[i]).numpy().tolist()
    heatmap_signal = result_lrp[i].numpy().tolist()

    for elem in range(len(neg_signal)):
        if heatmap_signal[elem] > average(heatmap_signal):
            neg_signal[elem] = np.nan

    for elem in range(len(pos_signal)):
        if heatmap_signal[elem] <= average(heatmap_signal):
            pos_signal[elem] = np.nan

    ax.plot(pos_signal, color='r')
    ax.plot(neg_signal, color='b')
# plt.savefig('/nas/home/amontali/tmp/save_example.pdf')
plt.show()
'''

'''
# Plot input signal with colors based on analysis


def average(lst):
    return sum(lst) / len(lst)


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


# pp = PdfPages('/nas/home/amontali/tmp/endtoend_plots.pdf')

for s in range(len(signals_for_analysis)):
    # Source considered for analysis
    print(sources_for_analysis[s])
    source = str(np.round(sources_for_analysis[s].numpy(), 2).tolist())[1:-1]

    # Signal considered for analysis [dimension for visualization: (8, 1280)]
    print(signals_for_analysis[s])
    input_sig = tf.transpose(signals_for_analysis[s])

    # Creating the analyzer
    analyzer = SimpleLRP(model)

    # Input signal dimension for analysis: (1280, 8, 1)
    input_signal = tf.expand_dims(signals_for_analysis[s], axis=0)

    # Applying the analyzer
    analysis = analyzer.analyze(input_signal)

    # Result dimension for visualization: (8, 1280)
    result_lrp = tf.transpose(tf.squeeze(analysis))
    
    # Take max and min value for mapping
    max_val = tf.reduce_max(result_lrp)
    min_val = tf.reduce_min(result_lrp)
    avg_heatmap_signal = tf.reduce_mean(result_lrp)

    # Plot input signal colored with heatmap values from analysis
    fig, axes = plt.subplots(ncols=1, nrows=8, figsize=(20, 8))
    for i, ax in enumerate(axes.flatten()):
        one_values = tf.identity(input_sig[i]).numpy().tolist()
        two_values = tf.identity(input_sig[i]).numpy().tolist()
        three_values = tf.identity(input_sig[i]).numpy().tolist()
        four_values = tf.identity(input_sig[i]).numpy().tolist()
        five_values = tf.identity(input_sig[i]).numpy().tolist()
        six_values = tf.identity(input_sig[i]).numpy().tolist()
        heatmap_signal = result_lrp[i].numpy().tolist()

        for elem in range(1280):
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

        ax.plot(one_values, color='maroon')
        ax.plot(two_values, color='red')
        ax.plot(three_values, color='orange')
        ax.plot(four_values, color='cyan')
        ax.plot(five_values, color='blue')
        ax.plot(six_values, color='darkblue')
    plt.suptitle(source)
    # pp.savefig()
    plt.show()

# pp.close()
'''

'''
# Heatmap plot

# Creating an analyzer
analyzer = SimpleLRP(model)

for i in range(len(signals_for_analysis)):
    # Source considered for analysis
    print(sources_for_analysis[i])
    source = str(np.round(sources_for_analysis[i].numpy(), 2).tolist())[1:-1]

    # Input signal dimension for analysis: (1280, 8, 1)
    input_signal = tf.expand_dims(signals_for_analysis[i], axis=0)

    # Applying the analyzer
    analysis = analyzer.analyze(input_signal)

    # Result dimension for visualization: (8, 1280)
    result_lrp = tf.transpose(tf.squeeze(analysis))

    # Input dimension for visualization: (8, 1280)
    input_signal = tf.transpose(signals_for_analysis[i])

    # Plot LRP result and input signal
    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(8, 5))

    heat_map = sb.heatmap(result_lrp, cmap="rocket", ax=ax, cbar=False)
    heat_map.invert_yaxis()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax.collections[0], cax=cax, orientation='vertical')

    heat_map_input = sb.heatmap(input_signal, cmap="rocket", ax=ax2, cbar=False)
    heat_map_input.invert_yaxis()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax2.collections[0], cax=cax, orientation='vertical')

    plt.sca(ax)
    plt.title("LRP result")
    plt.sca(ax2)
    plt.title("Input signal")
    plt.suptitle(source)
    plt.show()
'''

'''
# Plot intermediate output

test_dataset = create_endtoend_dataset(filepath=filename, buffer_size=2262, batch_size=100, train_size=1700,
                                       dataset_type='test')

model.summary()

# Index 0-2-3-5-7 (output convolutional layers)
output_layers = [0, 2, 3, 5, 7]

for i in output_layers:
    inter_output_model = tf.keras.Model(model.input, model.get_layer(index=i).output)
    inter_output = inter_output_model.predict(test_dataset, steps=6)
    print(inter_output.shape)

    # Select one batch
    inter_output = tf.transpose(inter_output[randrange(inter_output.shape[0])])
    print(inter_output.shape)
    inter_output_all_filters = tf.transpose(inter_output)
    print(inter_output_all_filters.shape)

    # Plot the intermediate output
    plt.figure()
    plt.plot(inter_output_all_filters)
    plt.title(i)
    plt.show()

    # Select one filter: the index depends on the convolutional layer (96 for 0,2 and 128 for 3,5,7)
    inter_output = inter_output[randrange(inter_output.shape[0])]
    print(inter_output.shape)

    # Plot the intermediate output
    plt.figure()
    plt.plot(inter_output)
    plt.title(i)
    plt.show()
'''

'''
# Remove layers and analyze

test_dataset = create_endtoend_dataset(filepath=filename, buffer_size=2262, batch_size=100, train_size=1700,
                                       dataset_type='test')
model.evaluate(test_dataset, steps=6)

model.summary()
print("Number of layers in the base model: ", len(model.layers))
model2 = tf.keras.models.Sequential()
for layer in model.layers[:-4]:
    model2.add(layer)
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(3, activation='linear'))
model2.summary()

analyzer = SimpleLRP(model2)
input_signal = tf.expand_dims(signals_for_analysis[0], axis=0)
analysis = analyzer.analyze(input_signal)
result_lrp = tf.transpose(tf.squeeze(analysis))

plt.figure()
heat_map = sb.heatmap(result_lrp, cmap="rocket")
heat_map.invert_yaxis()
plt.show()
'''

'''
# Old Version

filename_test_data = '/nas/home/amontali/tmp/endtoend_model_old.tfrecord'
checkpoint_filepath = '/nas/home/amontali/tmp/model_endtoend_old'


def read_tfrecord_test_data(serialized_example):
    feature_description = {
        'signal': tf.io.FixedLenFeature((), tf.string),
        'source': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    y_test = tf.io.parse_tensor(example['source'], out_type=tf.float64)
    x_test = tf.io.parse_tensor(example['signal'], out_type=tf.float64)
    return x_test, y_test


tfrecord_dataset = tf.data.TFRecordDataset(filename_test_data)
test_dataset = tfrecord_dataset.map(read_tfrecord_test_data)

signals_input_dataset = []
sources_input_dataset = []
signals_for_analysis = []
sources_for_analysis = []
count = 1
for data in test_dataset:
    signals_input_dataset.append(data[0])
    sources_input_dataset.append(data[1])
    if count == 541 or count == 526 or count == 493 or count == 391:
        signals_for_analysis.append(data[0])
        sources_for_analysis.append(data[1])
    count += 1

# Test signals for analysis
print(sources_for_analysis)

test_dataset = tf.data.Dataset.from_tensor_slices((signals_input_dataset, sources_input_dataset))
test_dataset = test_dataset.batch(100).repeat()

model = tf.keras.models.load_model(checkpoint_filepath)
model.evaluate(test_dataset, steps=6)


# Analysis with Innvestigate: Layer-Wise Relevance Propagation


class SimpleLRPRule(kgraph.ReverseMappingBase):

    # Basic LRP decomposition rule.

    def __init__(self, layer, state, bias=True):
        # Copy forward layer
        self._layer_wo_act = layer

    def apply(self, Xs, Ys, Rs, reverse_state):
        grad = ilayers.GradientWRT(len(Xs))

        # Get layers.
        Zs = kutils.apply(self._layer_wo_act, Xs)
        # Divide incoming relevance by the layers.
        tmp = [ilayers.SafeDivide()([a, b])
               for a, b in zip(Rs, Zs)]
        # Propagate the relevance to input neurons
        # using the gradient.
        tmp = iutils.to_list(grad(Xs+Zs+tmp))
        # Re-weight relevance with the input values.
        return [tf.keras.layers.Multiply()([a, b])
                for a, b in zip(Xs, tmp)]


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


# Creating an analyzer
analyzer = SimpleLRP(model)

for i in range(len(signals_for_analysis)):
    # Input signal dimension for analysis: (1280, 8, 1)
    print(sources_for_analysis[i])
    input_signal = tf.expand_dims(signals_for_analysis[i], axis=0)

    # Applying the analyzer
    analysis = analyzer.analyze(input_signal)

    # Result dimension for visualization: (8, 1280)
    result_lrp = tf.transpose(tf.squeeze(analysis))

    # Input dimension for visualization: (8, 1280)
    input_signal = tf.transpose(signals_for_analysis[i])

    # Plot LRP result and input signal
    source = ''
    if i == 0:
        source = '[0.77; 2.93; 0.92]'
    if i == 1:
        source = '[0.00; 2.93; 0.92]'
    if i == 2:
        source = '[0.77; 0.88; 0.92]'
    if i == 3:
        source = '[0.00; 0.88; 0.92]'

    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(8,5))

    heat_map = sb.heatmap(result_lrp, cmap="rocket", ax=ax, cbar=False)
    heat_map.invert_yaxis()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax.collections[0], cax=cax, orientation='vertical')

    heat_map_input = sb.heatmap(input_signal, cmap="rocket", ax=ax2, cbar=False)
    heat_map_input.invert_yaxis()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax2.collections[0], cax=cax, orientation='vertical')

    plt.sca(ax)
    plt.title("LRP result")
    plt.sca(ax2)
    plt.title("Input signal")
    plt.suptitle(source)
    plt.show()
'''