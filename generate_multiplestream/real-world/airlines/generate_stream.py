import numpy as np
import pandas as pd
from scipy.io import arff

data,_ = arff.loadarff('generate_multistream/real-world/airlines/airlines.arff')
df = pd.DataFrame(data)
df['Airline']=df['Airline'].str.decode('utf-8')
df['AirportFrom']=df['AirportFrom'].str.decode('utf-8')
df['AirportTo']=df['AirportTo'].str.decode('utf-8')
df['DayOfWeek']=df['DayOfWeek'].str.decode('utf-8')
df['Delay']=df['Delay'].str.decode('utf-8')

airline_map = {elem:index for index,elem in enumerate(set(df['Airline']))}
df['Airline']=df['Airline'].map(airline_map)

airport_map = {elem:index for index,elem in enumerate(set(df['AirportFrom']))}
df['AirportFrom']=df['AirportFrom'].map(airport_map)
df['AirportTo']=df['AirportTo'].map(airport_map)

stream_start = 20000
# stream_length can also be extended to 10000
stream_length = 5000
stream_end = stream_start + stream_length
df=df.iloc[stream_start:stream_end]
df.to_csv(f'log/real-world/airlines/stream_all.csv',index=0)

streams_attribute_name = [['Airline','Flight','AirportFrom','AirportTo'],['DayOfWeek','Time','Length']]
stream_label_name = 'Delay'

for index, stream_attribute_name in enumerate(streams_attribute_name):
    stream = df[stream_attribute_name].copy()
    stream[stream_label_name] = df[stream_label_name]
    stream.to_csv(f'log/real-world/airlines/stream_{index+1}.csv',index=0)