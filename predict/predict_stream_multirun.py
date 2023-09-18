import numpy as np
import pandas as pd

from predict_stream import *

from pathlib import Path

def predict_stream_multirun(base_dir, filename, model_names, fading_factors=np.array([0.99]), run_time=30):
    performance_names_base = ['mean_acc','mean_kappa','mean_recall','mean_f1']
    performance_names_all = []
    for model_name in model_names:
        performance_names_model = []
        for performance_name_base in performance_names_base:
            performance_names_model.append(f'{performance_name_base}_[{model_name}]')
        performance_names_all = performance_names_all + performance_names_model
    
    performance_all = np.zeros((run_time,len(performance_names_all)))
    for i in range(run_time):
        performance = predict_stream(base_dir, filename, model_names, fading_factors)
        performance_all[i,:] = performance.reshape(-1)
    
    df = pd.DataFrame(performance_all,columns=performance_names_all)
    
    model_name_all = print_model_names(model_names)
    output_folder = f'{base_dir}{model_name_all}/'
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)   
    output_file = f'{output_folder}{run_time}run_predict_{filename}'
    df.to_csv(output_file,index=0)
    
    
def predict_stream_with_passive_adapt_multirun(base_dir, filename, model_names, fading_factors=np.array([0.99]), run_time=30):
    performance_names_base = ['mean_acc','mean_kappa','mean_recall','mean_f1']
    performance_names_all = []
    for model_name in model_names:
        performance_names_model = []
        for performance_name_base in performance_names_base:
            performance_names_model.append(f'{performance_name_base}_[{model_name}]')
        performance_names_all = performance_names_all + performance_names_model
    
    performance_all = np.zeros((run_time,len(performance_names_all)))
    for i in range(run_time):
        performance = predict_stream_with_passive_adapt(base_dir, filename, model_names, fading_factors)
        performance_all[i,:] = performance.reshape(-1)
    
    df = pd.DataFrame(performance_all,columns=performance_names_all)
    
    model_name_all = print_model_names(model_names)
    output_folder = f'{base_dir}{model_name_all}_passive_adapt/'
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)   
    output_file = f'{output_folder}{run_time}run_predict_{filename}'
    df.to_csv(output_file,index=0)
    
def predict_stream_with_active_adapt_multirun(base_dir, filename, model_names, detect_drift_method_name, detected_drift_points, retrain_num = 200, reset=True, fading_factors=np.array([0.99]), run_time=30):
    performance_names_base = ['mean_acc','mean_kappa','mean_recall','mean_f1']
    performance_names_all = []
    for model_name in model_names:
        performance_names_model = []
        for performance_name_base in performance_names_base:
            performance_names_model.append(f'{performance_name_base}_[{model_name}]')
        performance_names_all = performance_names_all + performance_names_model
    
    performance_all = np.zeros((run_time,len(performance_names_all)))
    for i in range(run_time):
        performance = predict_stream_with_active_adapt(base_dir, filename, model_names, detect_drift_method_name, detected_drift_points, retrain_num = 200, reset=True, fading_factors=np.array([0.99]))
        performance_all[i,:] = performance.reshape(-1)
    
    df = pd.DataFrame(performance_all,columns=performance_names_all)
    
    model_name_all = print_model_names(model_names)
    if reset:
        output_folder = f'{base_dir}{model_name_all}_{detect_drift_method_name}_adapt_reset/'
    else:
        output_folder = f'{base_dir}{model_name_all}_{detect_drift_method_name}_adapt_noreset/'
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)   
    output_file = f'{output_folder}{run_time}run_predict_{filename}'
    df.to_csv(output_file,index=0)


if __name__ == '__main__':
    # base_dir = 'log/synthetic/dangerous_driving/sudden_correlation_drift/'
    # base_dir = 'log/synthetic/dangerous_driving/gradual_correlation_drift/'
    # base_dir = 'log/synthetic/dangerous_driving/incremental_correlation_drift/'
    
    # base_dir = 'log/synthetic/sin_function/sudden_correlation_drift/'
    # base_dir = 'log/synthetic/sin_function/gradual_correlation_drift/'
    # base_dir = 'log/synthetic/sin_function/incremental_correlation_drift/'
    
    # base_dir = 'log/real-world/poker_hand/'
    # base_dir = 'log/real-world/electricity/'
    base_dir = 'log/real-world/airlines/'
    
    filename = 'stream_all.csv'
    
    # model_names = ['NB','HT']
    
    model_names = ['NB','HAT','ARF']

    detect_drift_method_name = 'CDDM'
        
    # detected_drift_points = np.array([500])
    # detected_drift_points = np.array([410])
    # detected_drift_points = np.array([560])
    
    # detected_drift_points = np.array([500])
    # detected_drift_points = np.array([410])
    # detected_drift_points = np.array([230])
    
    # detected_drift_points = np.array([590.0,  890.0, 1270.0, 2010.0, 2550.0])
    # detected_drift_points = np.array([270.0, 1090.0, 1330.0, 1690.0, 2463.0])
    detected_drift_points = np.array([1050.0,4350.0])
    
    # predict_stream_multirun(base_dir,filename, model_names)
    # predict_stream_with_passive_adapt_multirun(base_dir,filename,model_names)
    predict_stream_with_active_adapt_multirun(base_dir, filename, model_names, detect_drift_method_name, detected_drift_points, reset=True)
    # predict_stream_with_active_adapt_multirun(base_dir, filename, model_names, detect_drift_method_name, detected_drift_points, reset=False)