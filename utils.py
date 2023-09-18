import pandas as pd
import numpy as np

from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.lazy import KNNClassifier
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.evaluation import EvaluatePrequential
from predict.evaluate import *

def save_labelled_stream(stream,filename):
    df = pd.DataFrame(stream)
    df[df.columns[-1]] = df[df.columns[-1]].astype('int')
    df.to_csv(filename,header=None,index=None)
    

def remove_comments(file):
    # delete comments (comment_cnt lines) in the file
    comment_cnt = 0
    fin = open(file,'r')
    lines = fin.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            comment_cnt +=1
        else:
            break
    lines_without_comments = ''.join(lines[comment_cnt:])
    fin.close()
    fout = open(file,'w')
    fout.write(lines_without_comments)
    fout.close()


if __name__ == '__main__':
    # Imports
    import numpy as np
    from skmultiflow.drift_detection import PageHinkley
    ph = PageHinkley()
    # Simulating a data stream as a normal distribution of 1's and 0's
    data_stream = np.random.randint(2, size=2000)
    # Changing the data concept from index 999 to 2000
    for i in range(999, 2000):
    # Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
        data_stream[i] = np.random.randint(4, high=8)
    for i in range(2000):
        ph.add_element(data_stream[i])
        if ph.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))