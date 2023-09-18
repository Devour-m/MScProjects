import numpy as np
import math

from utils import save_labelled_stream

# random seed
seed = 1
np.random.seed(seed)

# data stream hyperparameter
stream_length = 1000
drift_point = 500
frequency_before = np.array([0.26,0.34,0.27])
frequency_after = np.array([0.34,0.34,0.27])

def assign_magnitude(length):
    magnitude_norm = np.zeros(length)
    
    np.random.seed(None)
    while (magnitude_norm < round(1/length - 0.1,2)).any():
        magnitude = np.random.rand(length)
        magnitude_norm = np.round(magnitude/np.sum(magnitude),2)

    return magnitude_norm

def assign_label(x):
    y_prime = x
    # sigmoid function, where 0.5 is a hyperparameter that determines the decision boundary
    y = 1/(1+math.exp(-y_prime))
    if y > 0.5:
        return 1
    else:
        return 0

index_list = np.array([index for index in range(stream_length)])

# stream 1: magnitude_1 is [0.26, 0.34, 0.39]
stream_1 = np.zeros((stream_length,2))
magnitude_1 = np.array([0.26, 0.34, 0.39])

for index in range(stream_length):
    if index < drift_point:
        current_frequency = frequency_before
    else:
        current_frequency = frequency_after
    stream_1[index,0] = np.sum(magnitude_1 * np.array([math.cos(current_frequency[0] * index * math.pi/2), math.sin(current_frequency[0] * index * math.pi/2), math.cos(current_frequency[0] * index * math.pi/2)]))
    stream_1[index,1] = assign_label(stream_1[index,0])
    
save_labelled_stream(stream_1, 'log/synthetic/sin_function/sudden_correlation_drift/stream_1.csv')

# stream 2: magnitude_2 is [0.4, 0.34, 0.27]
stream_2 = np.zeros((stream_length,2))
magnitude_2 = np.array([0.4, 0.34, 0.27])

for index in range(stream_length):
    if index < drift_point:
        current_frequency = frequency_before
    else:
        current_frequency = frequency_after
    stream_2[index,0] = np.sum(magnitude_2 * np.array([math.sin(current_frequency[1] * index * math.pi/2), math.cos(current_frequency[1] * index * math.pi/2), math.sin(current_frequency[1] * index * math.pi/2)]))
    stream_2[index,1] = assign_label(stream_2[index,0])
    
save_labelled_stream(stream_2, 'log/synthetic/sin_function/sudden_correlation_drift/stream_2.csv')

# stream 3: magnitude_3 is [0.33, 0.29, 0.27]
stream_3 = np.zeros((stream_length,2))
magnitude_3 = np.array([0.33, 0.29, 0.27])

for index in range(stream_length):
    if index < drift_point:
        current_frequency = frequency_before
    else:
        current_frequency = frequency_after
    stream_3[index,0] = np.sum(magnitude_3 * np.array([math.cos(current_frequency[2] * index * math.pi/2), math.cos(current_frequency[2] * index * math.pi/2), math.sin(current_frequency[2] * index * math.pi/2)]))
    stream_3[index,1] = assign_label(stream_3[index,0])
    
save_labelled_stream(stream_3, 'log/synthetic/sin_function/sudden_correlation_drift/stream_3.csv')

# stream_1 and stream_2 and stream_3
stream_all = np.zeros((stream_length,4))
stream_all[:,:-1] = np.concatenate(
    (stream_1[:, 0].reshape(-1, 1), stream_2[:, 0].reshape(-1, 1),stream_3[:, 0].reshape(-1, 1)), axis=1)

for index in range(stream_all.shape[0]):
    stream_all[index,-1] = assign_label(np.mean(stream_all[index,:-1]))
save_labelled_stream(stream_all,'log/synthetic/sin_function/sudden_correlation_drift/stream_all.csv')