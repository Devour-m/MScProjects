import numpy as np
import pandas as pd
from scipy.stats import ranksums

from model import print_model_names

def evaluate_mean_acc_multirun(base_dir, filename, model_names, adapt_methods, run_time=30):            
    mean_acc_multirun = list()     

    model_name_all = print_model_names(model_names)
    for adapt_method in adapt_methods:
        if adapt_method == '':
            file = f'{base_dir}{model_name_all}/{run_time}run_predict_{filename}'
        else:
            file = f'{base_dir}{model_name_all}_{adapt_method}/{run_time}run_predict_{filename}'
        df = pd.read_csv(file)
        
        mean_acc = list()
        
        for model_name in model_names:
            mean_acc_model = np.array(df[f'mean_acc_[{model_name}]'])
            mean_acc_model_mean = np.mean(mean_acc_model)
            mean_acc_model_std = np.std(mean_acc_model)
            mean_acc_model_output = f'{(100*mean_acc_model_mean):.2f}\pm {(100*mean_acc_model_std):.2f}'
            mean_acc.append(mean_acc_model_output)
            
        mean_acc_multirun.append(mean_acc)
    
    df_rows = ['no_adapt' if adapt_method in [''] else adapt_method for adapt_method in adapt_methods]
    df_cols = model_names
    df = pd.DataFrame(mean_acc_multirun,index=df_rows,columns=df_cols)
    
    output_file = f'{base_dir}{run_time}run_predict_{filename}'
    output_file_without_abbr = output_file.split('.')[0]
    df.to_excel(f'{output_file_without_abbr}.xlsx')
      
def compare_signifcance(x,y):
    res = ranksums(x,y)
    if res.pvalue >= 0.05:
        return 0
    elif np.mean(x) < np.mean(y):
        return -1
    elif np.mean(x) == np.mean(y):
        return 0
    else:
        return 1
 
def compare_mean_acc_multirun(base_dir,filename,model_names,adapt_methods=['CDDM_adapt_reset','CDDM_adapt_noreset'],run_time=30):
    model_name_all = print_model_names(model_names)

    for model_name in model_names:
        acc_list = []
        acc_list_rank = np.zeros(len(adapt_methods))
        
        for adapt_method in adapt_methods:
            if adapt_method == '':
                file = f'{base_dir}{model_name_all}/{run_time}run_predict_{filename}'
            else:
                file = f'{base_dir}{model_name_all}_{adapt_method}/{run_time}run_predict_{filename}'
            df = pd.read_csv(file)
            acc_list.append(np.array(df[f'mean_acc_[{model_name}]']))
            
        for i  in range(len(adapt_methods)):
            for j in range(i,len(adapt_methods)):
                comparison = compare_signifcance(acc_list[i],acc_list[j])
                acc_list_rank[i] = acc_list_rank[i] + comparison
                acc_list_rank[j] = acc_list_rank[j] - comparison
        
        if np.max(acc_list_rank) > 0:        
            best_adapt_method = adapt_methods[np.argmax(acc_list_rank)]
        else:
            best_adapt_method = None
            
        print(f'{model_name}: best adapt method is {best_adapt_method}')
        
        
                
  
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
    
    adapt_methods = ['','CDDM_adapt_reset','CDDM_adapt_noreset']
    
    # evaluate_mean_acc_multirun(base_dir, filename, model_names, adapt_methods)
    compare_mean_acc_multirun(base_dir,filename,model_names)