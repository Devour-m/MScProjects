import os
import numpy as np
import pandas as pd
import matlab.engine
import logging
from pathlib import Path

from correlation import correlation_pairwise, correlation_mean
from skmultiflow.drift_detection import PageHinkley


def read_streams(base_dir,header):
    streams = list()
    stream_all = None
    
    base_dir_list = os.listdir(base_dir)
    for file in base_dir_list:
        file_abbr = file.split('.')[-1]
        file_dir = os.path.join(base_dir, file)
        if (file.startswith('stream')) and (file_abbr=='csv'):
            if header is None:
                df = pd.read_csv(file_dir, header=None)
            else:
                df = pd.read_csv(file_dir)
            if file.split('.')[0].endswith('all'):
                stream_all = np.array(df)
            else:
                streams.append(np.array(df))
            
    return streams, stream_all
        

# HCDT only detects P(X) change, by checking the mean and variance of X. HCDT cannot track the P(y|X) change
def detect_single_stream(stream, has_label, drift_detection_method):
    eng = matlab.engine.start_matlab()
    eng.cd('../../material/algorithm/single-data-stream concept drift detection/main/',nargout=0)
    
    if drift_detection_method == 'HCDT':
        if has_label:
            stream = stream[:,:-1]
        if stream.ndim == 1:
            stream = stream.reshape(-1,1)
        detected_drift_points = eng.HCDT(matlab.double(stream.tolist()))
    elif (drift_detection_method == 'HLFR') and has_label:
        detected_drift_points = eng.HLFR(matlab.double(stream.tolist()))
    elif drift_detection_method == 'CUSUM':
        detected_drift_points = np.array([])
        if has_label:
            stream = stream[:,:-1]
            
        ph = PageHinkley()
        for i in range(stream.shape[0]):
            ph.add_element(stream[i,:])
            if ph.detected_change():
                detected_drift_points = np.append(detected_drift_points, i)
    else:
        detected_drift_points = None
    eng.quit()
    return detected_drift_points

corrs_window_size = 50

single_stream_drift_detection_methods = ['HCDT','HLFR']

if __name__ == '__main__':
    base_dir = 'log/synthetic/dangerous_driving/gradual_correlation_drift/'
    drift_detection_method = 'CDDM'
    
    output_folder = f'{base_dir}detect_result/'
    path = Path(output_folder)    
    path.mkdir(parents=True, exist_ok=True)
    output_file = f'{output_folder}detect_stream_{drift_detection_method}.log'
    logging.basicConfig(level=logging.INFO, filename=output_file, format='%(message)s', filemode='w')
    
    streams, stream_all = read_streams(base_dir,header=np.array([]))  
    
    # calculate correlation
    # corrs = correlation_mean(streams,correlation_type='spearman',window_size=corrs_window_size,save_file=True,output_dir=f'{base_dir}correlation.csv')
    corrs = correlation_pairwise(streams[1],streams[2],correlation_type='cross',window_size=corrs_window_size,save_file=True,output_dir=f'{base_dir}correlation.csv')
    
    logging.debug(f'Correlation stream: {corrs}')
    
    if drift_detection_method in single_stream_drift_detection_methods:
        for i,stream in enumerate(streams):
            stream_detection = detect_single_stream(stream, True, drift_detection_method)
            logging.info(f'Stream {i+1} concept drift detection ({drift_detection_method}): {stream_detection}')
            
    if drift_detection_method == 'CDDM':
        try:
            corrs_detection = np.asarray(detect_single_stream(corrs, False, 'HCDT')) + corrs_window_size
        except:
            corrs_detection = np.array([])
        logging.info(f'Correlation drift detection (CDDM): {corrs_detection}')
        
    if drift_detection_method == 'CDDM-CUSUM':
        try:
            corrs_detection = np.asarray(detect_single_stream(corrs, False, 'CUSUM')) + corrs_window_size
        except:
            corrs_detection = np.array([])
        logging.info(f'Correlation drift detection (CDDM-CUSUM): {corrs_detection}')