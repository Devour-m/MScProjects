import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_correlation(base_dir, filename, drift_points, drift_type='sudden',sliding_window_size=50):
    df = pd.read_csv(f'{base_dir}{filename}',header=None)
    corr = np.array(df[0])
    time = np.array([i for i in range(sliding_window_size, sliding_window_size + corr.shape[0])])

    plt.plot(time,corr)
    plt.xlabel('Sample number')
    plt.ylabel('Correlation')
    
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
        drift_range = np.array([drift_points[0], sliding_window_size + corr.shape[0]])
        plt.axvspan(drift_range[0],drift_range[1],color='tab:red',alpha=0.3)
           
    middle_dir = base_dir.split('/',1)[-1]
    filename_without_abbr = filename.split('.')[0]
    plt.savefig(f'result/{middle_dir}{filename_without_abbr}.png',dpi=600)
    # plt.show()
    plt.close()
    
if __name__ == '__main__':
    base_dir = 'log/synthetic/dangerous_driving/gradual_correlation_drift/'
    filename = 'correlation.csv' 
    drift_type = 'gradual'
    
    # drift_points = np.array([500])
    drift_points = np.array([400,600])
    # drift_points = np.array([200])
    
    
    plot_correlation(base_dir,filename,drift_points,drift_type)