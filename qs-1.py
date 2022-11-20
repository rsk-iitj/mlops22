from unittest import result
from sklearn import datasets, metrics, svm
from sklearn import tree
import statistics
import pandas as pd
import numpy as np
from joblib import dump
# Starts actual execution

from sklearn.model_selection import train_test_split

digits = datasets.load_digits()


def data_preprocess(data):
    # flatten the images
    n_samples = len(data.images)
    x = data.images.reshape((n_samples, -1))
    return x,data.target

data, label = data_preprocess(digits)

def train_test_split_test(random_st,test_sz):
    
    
    x_train, x_test, y_train, x_test = train_test_split(
        data, label, random_state = random_st, test_size=test_sz,shuffle=True
    )
    
    return x_train, x_test,y_train,x_test



def test_random_split_same():
    random_state_1 = 40
    random_state_2 = 40
    test_size = 20
    x_train1, x_test1, y_train1, y_test1 = train_test_split_test(test_size,random_state_1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split_test(test_size,random_state_2)
    assert  np.array_equal(x_train1,x_train2)
    assert  np.array_equal(y_train1 ,y_train2)
    assert  np.array_equal(x_test1 , x_test2)
    assert  np.array_equal(y_test1 ,y_test2)
    
    
    
def test_random_split_not_same():
    random_state_1 = 40
    random_state_2 = 30
    test_size = 20
    x_train1, x_test1, y_train1, y_test1 = train_test_split_test(test_size,random_state_1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split_test(test_size,random_state_2)
    assert not np.array_equal(x_train1, x_train2)
    assert not np.array_equal(y_train1, y_train2)
    assert not np.array_equal(x_test1, x_test2)
    assert not np.array_equal(y_test1, y_test2)