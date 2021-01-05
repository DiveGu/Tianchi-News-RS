from tensorflow.keras.layers import *
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import * 
import numpy as np
import pandas as pd


#x={1,2,3,}
#print(type(x))
#print(x)

#x=[1,2]
#a,b=x
#print(b)


x=np.array([[3,6,1,19]])
#x.sort(axis=1)
#print(x[:1,:2])
#print(x.shape)
idx=x.argsort(axis=1)[0][::-1][:2]
print(idx)
x.sort(axis=1)
sim=x[0][-2:]
print(sim)