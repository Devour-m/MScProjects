import numpy as np
import pandas as pd

from skmultiflow.data.file_stream import FileStream
from skmultiflow.evaluation import EvaluatePrequential

from model import init_models, print_model_names
from evaluate import evaluate, fading_accuracy, get_evaluation_mean_performance

from pathlib import Path
from utils import remove_comments


def predict_stream(base_dir, filename, model_names, fading_factors=np.array([0.99])):
    model_name_all = print_model_names(model_names)
    
    output_folder = f'{base_dir}{model_name_all}'
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)   
    output_file = f'{output_folder}/predict_{filename}'
    
    stream = FileStream(f'{base_dir}{filename}')
    models = init_models(model_names)
    evaluator = EvaluatePrequential(n_wait=1, max_samples = 10000, pretrain_size=200, metrics=['accuracy','kappa','recall','f1'],show_plot=False,output_file=output_file)
    evaluate(evaluator, stream, models, model_names, need_adapt=False)
    
    remove_comments(output_file)
    
    # add mean_accuracy with fading factor
    df = pd.read_csv(output_file)
    for model_name in model_names:
        cur_accuracy = df[f'current_acc_[{model_name}]']
        for fading_factor in fading_factors:
            df[f'mean_acc_{fading_factor}_fading_[{model_name}]'] = fading_accuracy(cur_accuracy,fading_factor)
    df.to_csv(output_file,index=0)    
    
    return get_evaluation_mean_performance(evaluator)
    
def predict_stream_with_passive_adapt(base_dir, filename, model_names, fading_factors=np.array([0.99])):
    model_name_all = print_model_names(model_names)
    
    output_folder = f'{base_dir}{model_name_all}_passive_adapt/'
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    output_file = f'{output_folder}predict_{filename}'
    
    stream = FileStream(f'{base_dir}{filename}')
    models = init_models(model_names)
    evaluator = EvaluatePrequential(n_wait=1, max_samples = 10000, pretrain_size=200, metrics=['accuracy','kappa','recall','f1'],show_plot=False,output_file=output_file)
    evaluator.evaluate(stream=stream, model=models,model_names=model_names)
    
    remove_comments(output_file)
    
    # add mean_accuracy with fading factor
    df = pd.read_csv(output_file)
    for model_name in model_names:
        cur_accuracy = df[f'current_acc_[{model_name}]']
        for fading_factor in fading_factors:
            df[f'mean_acc_{fading_factor}_fading_[{model_name}]'] = fading_accuracy(cur_accuracy,fading_factor)
    df.to_csv(output_file, index=0)
    
    return get_evaluation_mean_performance(evaluator)

def predict_stream_with_active_adapt(base_dir, filename, model_names, detect_drift_method_name, detected_drift_points, retrain_num = 200, reset=True, fading_factors=np.array([0.99])):
    model_name_all = print_model_names(model_names)
    
    if reset:
        output_folder = f'{base_dir}{model_name_all}_{detect_drift_method_name}_adapt_reset/'
    else:
        output_folder = f'{base_dir}{model_name_all}_{detect_drift_method_name}_adapt_noreset/'
    path = Path(output_folder)
    path.mkdir(parents=True, exist_ok=True)
    output_file = f'{output_folder}predict_{filename}'
    
    stream = FileStream(f'{base_dir}{filename}')
    models = init_models(model_names)
    
    evaluator = EvaluatePrequential(n_wait=1, max_samples = 10000, pretrain_size=200, metrics=['accuracy','kappa','recall','f1'], show_plot=False,output_file=output_file)
    
    evaluate(evaluator=evaluator, stream=stream,model=models,model_names=model_names, need_adapt=True, detected_drift_points=detected_drift_points, retrain_num = retrain_num, reset=reset)
    
    remove_comments(output_file)
    
    # add mean_accuracy with fading factor
    df = pd.read_csv(output_file)
    for model_name in model_names:
        cur_accuracy = df[f'current_acc_[{model_name}]']
        for fading_factor in fading_factors:
            df[f'mean_acc_{fading_factor}_fading_[{model_name}]'] = fading_accuracy(cur_accuracy,fading_factor)
    df.to_csv(output_file, index=0)

    return get_evaluation_mean_performance(evaluator)
    
if __name__ == '__main__':
    base_dir = 'log/synthetic/dangerous_driving/sudden_correlation_drift/'
    # base_dir = 'log/synthetic/dangerous_driving/gradual_correlation_drift/'
    # base_dir = 'log/synthetic/dangerous_driving/incremental_correlation_drift/'
    
    # base_dir = 'log/synthetic/sin_function/sudden_correlation_drift/'
    # base_dir = 'log/synthetic/sin_function/gradual_correlation_drift/'
    # base_dir = 'log/synthetic/sin_function/incremental_correlation_drift/'
    
    # base_dir = 'log/real-world/poker_hand/'
    # base_dir = 'log/real-world/electricity/'
    # base_dir = 'log/real-world/airlines/'
    
    filename = 'stream_all.csv'
    
    # model_names = ['NB','HT']
    
    model_names = ['NB','HAT','ARF']

    detect_drift_method_name = 'CDDM'
        
    detected_drift_points = np.array([500])
    # detected_drift_points = np.array([410])
    # detected_drift_points = np.array([560])
    
    # detected_drift_points = np.array([500])
    # detected_drift_points = np.array([410])
    # detected_drift_points = np.array([230])
    
    # detected_drift_points = np.array([590.0,  890.0, 1270.0, 2010.0, 2550.0])
    # detected_drift_points = np.array([270.0, 1090.0, 1330.0, 1690.0, 2463.0])
    # detected_drift_points = np.array([1050.0,4350.0])
    
    predict_stream(base_dir,filename, model_names)
    predict_stream_with_passive_adapt(base_dir,filename,model_names)
    predict_stream_with_active_adapt(base_dir, filename, model_names, detect_drift_method_name, detected_drift_points, reset=True)
    predict_stream_with_active_adapt(base_dir, filename, model_names, detect_drift_method_name, detected_drift_points, reset=False)