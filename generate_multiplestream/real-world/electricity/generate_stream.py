import numpy as np
import pandas as pd
from scipy.io import arff

'''
class: 0-->'UP'; 1-->'DOWN'
'''

data, _ = arff.loadarff('generate_multistream/real-world/electricity/elecNormNew.arff')
df = pd.DataFrame(data)
df['day']=df['day'].str.decode('utf-8').astype('int')
df.loc[df['class']==b'UP','class']='0'
df.loc[df['class']==b'DOWN','class']='1'

stream_start = 1000
stream_length = 5000
stream_end = stream_start + stream_length
df=df.iloc[stream_start:stream_end]
df.to_csv(f'log/real-world/electricity/stream_all.csv',index=0)


streams_attribute_name = [['date','day','period','transfer'],['nswprice','nswdemand'],['vicprice','vicdemand']]
stream_label_name = 'class'

for index, stream_attribute_name in enumerate(streams_attribute_name):
    stream = df[stream_attribute_name].copy()
    stream[stream_label_name] = df[stream_label_name]
    stream.to_csv(f'log/real-world/electricity/stream_{index+1}.csv',index=0)