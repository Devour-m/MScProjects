import numpy as np
import pandas as pd
from scipy.io import arff

data_length = 10000

data,_ = arff.loadarff('generate_multistream/real-world/poker_hand/poker-lsn.arff')
df = pd.DataFrame(data)
df['s1'] = df['s1'].str.decode('utf-8').astype('int')
df['r1']=df['r1'].astype('int')
df['s2'] = df['s2'].str.decode('utf-8').astype('int')
df['r2']=df['r2'].astype('int')
df['s3'] = df['s3'].str.decode('utf-8').astype('int')
df['r3']=df['r3'].astype('int')
df['s4'] = df['s4'].str.decode('utf-8').astype('int')
df['r4']=df['r4'].astype('int')
df['s5'] = df['s5'].str.decode('utf-8').astype('int')
df['r5']=df['r5'].astype('int')
df['class'] = df['class'].str.decode('utf-8')

stream_start = 30000
stream_length = 5000
stream_end = stream_start + stream_length
df=df.iloc[stream_start:stream_end]
df.to_csv(f'log/real-world/poker_hand/stream_all.csv',index=0)


streams_attribute_name = [['s1','r1'],['s2','r2'],['s3','r3'],['s4','r4'],['s5','r5']]
stream_label_name = 'class'

for index, stream_attribute_name in enumerate(streams_attribute_name):
    stream = df[stream_attribute_name].copy()
    stream[stream_label_name] = df[stream_label_name]
    stream.to_csv(f'log/real-world/poker_hand/stream_{index+1}.csv',index=0)