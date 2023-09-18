import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_single_stream(base_dir, filename, drift_points, drift_type='sudden'):
    df = pd.read_csv(f'{base_dir}{filename}',header=None)
    df = df.reset_index()

    df_negative = df[df[1]==0]
    df_positive = df[df[1]==1]
    plt.scatter(x=df_negative['index'],y=df_negative[0],label='Low',color='tab:blue')
    plt.scatter(x=df_positive['index'],y=df_positive[0],label='High',color='tab:green')
    plt.legend(loc='upper right',title='Classes')

    plt.xlabel('Timestamp')
    plt.ylabel('Data value')
    
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
    plt.savefig(f'result/{middle_dir}{filename_without_abbr}.png',dpi=600)
    # plt.show()
    plt.close()
    
if __name__ == '__main__':
    base_dir = 'log/synthetic/sin_function/incremental_correlation_drift/'
    filename_list = ['stream_1.csv','stream_2.csv','stream_3.csv']
    drift_type = 'incremental'
    
    # drift_points = np.array([500])
    # drift_points = np.array([400,600])
    drift_points = np.array([200])
    
    for filename in filename_list:
        plot_single_stream(base_dir, filename, drift_points, drift_type)