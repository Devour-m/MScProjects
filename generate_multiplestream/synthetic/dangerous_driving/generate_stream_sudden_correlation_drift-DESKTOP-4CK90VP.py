import numpy as np
from scipy.stats import bernoulli

from utils import save_labelled_stream

# random seed
seed = 1
np.random.seed(seed)

# data stream hyperparameter
stream_length = 1000
drift_point = 500

# stream_1: road busy level
stream_1 = np.zeros((stream_length,2))
stream_1[:,1] = bernoulli.rvs(size=stream_length, p=0.5)

for index in range(stream_1.shape[0]):
    instance_label = stream_1[index,:]
    
    np.random.seed(index)
    if instance_label[1] == 0:
        stream_1[index,0] = np.random.uniform(0,1)
    else:
        stream_1[index,0] = np.random.uniform(1,2)

save_labelled_stream(stream_1, 'log/synthetic/dangerous_driving/sudden_correlation_drift/stream_1.csv')


# stream_2: driver focus level
stream_2 = np.zeros((stream_length,2))

for index in range(stream_2.shape[0]):
    instance_label = stream_1[index,:]
    
    np.random.seed(index)
    if instance_label[1] == 1:
        stream_2[index,:] = np.array([np.random.uniform(0,2),0])
    else:
        stream_2[index,:] = np.array([np.random.uniform(2,4),1])


save_labelled_stream(stream_2,'log/synthetic/dangerous_driving/sudden_correlation_drift/stream_2.csv')


# stream_3: speed
stream_3 = np.zeros((stream_length,2))

for index in range(stream_3.shape[0]):
    instance_label = stream_2[index,:]
    
    np.random.seed(index)
    
    if index < drift_point:
        if instance_label[1] == 0:
            stream_3[index,:] = np.array([np.random.uniform(3,6),1])
        else:
            stream_3[index,:] = np.array([np.random.uniform(0,3),0])
    else:
        if instance_label[1] == 0:
            stream_3[index,:] = np.array([np.random.uniform(0,3),0])
        else:
            stream_3[index,:] = np.array([np.random.uniform(3,6),1])

save_labelled_stream(stream_3,'log/synthetic/dangerous_driving/sudden_correlation_drift/stream_3.csv')

# stream_all
stream_all = np.zeros((stream_length,4))
stream_all[:,:-1] = np.concatenate(
    (stream_1[:, 0].reshape(-1, 1), stream_2[:, 0].reshape(-1, 1),stream_3[:, 0].reshape(-1, 1)), axis=1)

for index in range(stream_all.shape[0]):
    instance_label = np.array([stream_1[index,-1],stream_2[index,-1],stream_3[index,-1]])
    
    if np.sum(instance_label) <= 1:
        stream_all[index,-1] = 0
    else:
        stream_all[index,-1] = 1 
        
save_labelled_stream(stream_all,'log/synthetic/dangerous_driving/sudden_correlation_drift/stream_all.csv')
