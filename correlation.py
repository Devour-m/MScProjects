import numpy as np
import scipy.signal as signal
from scipy.stats import pearsonr, spearmanr

# suppose 1-D stream_1 and 1-D stream_2 has the same size
def correlation_pairwise_univariate(stream_x1, stream_x2, window_size=50,correlation_type='cross',save_file=True,output_dir=''):
    corrs = np.array([])
    for i in range(window_size, stream_x1.shape[0]):
        stream_1_slot = stream_x1[i-window_size:i]
        stream_2_slot = stream_x2[i-window_size:i]
        if correlation_type == 'cross':
            corr = signal.correlate(stream_1_slot, stream_2_slot, mode='full')
            corr_lag_0 = corr[corr.shape[0]//2]/window_size
            corrs = np.append(corrs, corr_lag_0)
        elif correlation_type == 'pearsonr':
            res = pearsonr(stream_1_slot,stream_2_slot)
            corrs = np.append(corrs,res.statistic)
        elif correlation_type == 'spearman':
            res = spearmanr(stream_1_slot,stream_2_slot)
            corrs = np.append(corrs,res.correlation)
        else:
            pass
    
    if save_file:
        np.savetxt(output_dir,corrs,delimiter=',')
    return corrs


# suppose 2-D stream_1 and 1-D stream_2 has the same size
# Each row represents a data instance, each row represents an attribute (one dimension) of data instances
def correlation_pairwise_multivariate(stream_x1,stream_x2, correlation_type = 'cross', window_size=50,save_file=True,output_dir=''):
    corrs = np.array([])
    for t in range(window_size, stream_x1.shape[0]):
        stream_1_slot = stream_x1[t-window_size:t,:]
        stream_2_slot = stream_x2[t-window_size:t,:]
        
        corrs_pairdim = np.array([])
        for i in range(stream_x1.shape[1]):
            for j in range(stream_x2.shape[1]):
                stream_1_slot_univariate = stream_1_slot[:,i]
                stream_2_slot_univariate = stream_2_slot[:,j]
                if correlation_type == 'cross':
                    corr = signal.correlate(stream_1_slot_univariate, stream_2_slot_univariate, mode='full')
                    corr_lag_0 = corr[corr.shape[0]//2]/window_size
                    corrs_pairdim = np.append(corrs_pairdim,corr_lag_0)
                elif correlation_type == 'pearson':
                    res = pearsonr(stream_1_slot_univariate, stream_2_slot_univariate)
                    corrs_pairdim = np.append(corrs_pairdim, res.statistic)
                elif correlation_type == 'spearman':
                    res = spearmanr(stream_1_slot_univariate, stream_2_slot_univariate)
                    corrs_pairdim = np.append(corrs_pairdim, res.correlation)
        corrs = np.append(corrs,np.mean(corrs_pairdim))
        
    if save_file:
        np.savetxt(output_dir,corrs,delimiter=',')
    return corrs

def correlation_pairwise(stream_1,stream_2,correlation_type='cross',window_size=50,save_file=True,output_dir=''):
    stream_x1 = stream_1[:,:-1]
    stream_x2 = stream_2[:,:-1]
    if (stream_x1.ndim == 1) & (stream_x2.ndim==1):
        return correlation_pairwise_univariate(stream_x1,stream_x2,correlation_type,window_size,save_file,output_dir)
    if (stream_x1.ndim ==2) & (stream_x2.ndim==2):
        return correlation_pairwise_multivariate(stream_x1,stream_x2,correlation_type,window_size,save_file,output_dir)
    else:
        return None 
    
def correlation_mean(streams, correlation_type='cross',window_size=50, save_file=True, output_dir = ''):
    stream_length = streams[0].shape[0]
    stream_num = len(streams)
    
    corrs = np.zeros(stream_length-window_size)
    cnt = 0
    for i in range(stream_num):
        stream_1 = streams[i]
        for j in range(i+1, stream_num):
            stream_2 = streams[j]
            corrs = corrs + correlation_pairwise(stream_1, stream_2, correlation_type, window_size, save_file=False)
            cnt = cnt + 1     
    corrs = corrs/cnt
    
    if save_file:
        np.savetxt(output_dir,corrs,delimiter=',')
    return corrs
    