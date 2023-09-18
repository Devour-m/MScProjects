import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from predict.model import print_model_names


def plot_predict_stream(base_dir, model_names_all, filename, model_name, acc_fading_factor, drift_points, drift_type='sudden', adapt_methods = ['']):   

    # accuracy
    for adapt_method in adapt_methods:
        if adapt_method == '':
            model_adapt_name = model_name
            df = pd.read_csv(f'{base_dir}{model_names_all}/{filename}')
        else:
            model_adapt_name = f'{model_name}_{adapt_method}'
            df = pd.read_csv(f'{base_dir}{model_names_all}_{adapt_method}/{filename}')
        
        if acc_fading_factor == 1:
            mean_acc = df[f'mean_acc_[{model_name}]']
        else:
            mean_acc = df[f'mean_acc_{acc_fading_factor}_fading_[{model_name}]']    
                              
        plt.plot(df['id'],mean_acc, label = model_adapt_name)
        
    if len(adapt_methods) > 1:
        plt.legend(loc='upper right')
    plt.xlabel('Sample number')
    plt.ylabel('Classification accuracy')

    # drift region
    if drift_type =='sudden':
        for drift_point in drift_points:
            plt.axvline(drift_point,color='red',linestyle='--')
    elif drift_type == 'gradual':
        drift_range = np.zeros(2)
        for index, drift_point in enumerate(drift_points):
            drift_range[index % 2] = drift_point
            if index % 2==1:
                plt.axvspan(drift_range[0],drift_range[1],color='tab:red',alpha=0.3)
    elif drift_type == 'incremental':
        drift_range = np.array([drift_points[0], len(df)])
        plt.axvspan(drift_range[0],drift_range[1],color='tab:red',alpha=0.3)
    
    
    middle_dir = base_dir.split('/',1)[-1]
    filename_without_abbr = filename.split('.')[0]
        
    if len(adapt_methods) == 1:
        if adapt_methods[0] == '':
            output_folder = f'result/{middle_dir}{model_name}/'
        else:
            output_folder = f'result/{middle_dir}{model_name}_{adapt_methods[0]}/'
    else:
        output_folder = f'result/{middle_dir}{model_name}_all/'
        
    output_folder = f'{output_folder}fading_{acc_fading_factor}/'
    output_file = f'{output_folder}{filename_without_abbr}.png'
               
    path = Path(output_folder)    
    path.mkdir(parents=True, exist_ok=True)       
               
    plt.savefig(output_file,dpi=600)
    # plt.show()
    plt.close()
    
if __name__ == '__main__':
    # base_dir = 'log/synthetic/dangerous_driving/gradual_correlation_drift/'
    base_dir = 'log/synthetic/sin_function/incremental_correlation_drift/'
    
    # base_dir = 'log/real-world/airlines/'
    # base_dir = 'log/real-world/electricity/'
    # base_dir = 'log/real-world/poker_hand/'
    
    filename_list = ['predict_stream_1.csv','predict_stream_2.csv','predict_stream_3.csv','predict_stream_all.csv']
    
    # filename_list = ['predict_stream_1.csv','predict_stream_2.csv','predict_stream_3.csv','predict_stream_4.csv', 'predict_stream_5.csv','predict_stream_6.csv','predict_stream_7.csv','predict_stream_8.csv','predict_stream_9.csv']
    # filename_list = ['predict_stream_1.csv','predict_stream_2.csv','predict_stream_3.csv','predict_stream_4.csv', 'predict_stream_5.csv','predict_stream_6.csv','predict_stream_7.csv']
    # filename_list = ['predict_stream_1.csv','predict_stream_2.csv','predict_stream_3.csv','predict_stream_4.csv','predict_stream_all.csv']
    
    acc_fading_factors = np.array([1,0.99,0.999])
    
    drift_type = 'incremental'
    
    # drift_points = np.array([5000])
    # drift_points = np.array([4000,6000])
    drift_points = np.array([2000])
    
    # drift_points = np.array([420.0,800.0])
    # drift_points = np.array([1180.0,1660.0])
    # drift_points = np.array([3870,5710,8196,9476])


    adapt_methods = ['','passive_adapt','CDDM_adapt_reset','CDDM_adapt_noreset']
    
    model_names = ['NB','HT']
    model_names_all = print_model_names(['NB','HT'])
    
    for filename in filename_list:
        for model_name in model_names:
            for acc_fading_factor in acc_fading_factors:
                for adapt_method in adapt_methods:
                    plot_predict_stream(base_dir, model_names_all, filename, model_name, acc_fading_factor, drift_points, drift_type, [adapt_method])
                plot_predict_stream(base_dir, model_names_all, filename, model_name, acc_fading_factor, drift_points, drift_type, adapt_methods)