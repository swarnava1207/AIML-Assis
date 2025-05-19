import sys
import os
import numpy as np


# Add the 'oracle' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'oracle'))
import oracle 
data = oracle.q2_train_test_emnist(23746,"EMNIST/emnist-balanced-train.csv","EMNIST/emnist-balanced-test.csv")
print(data[0].shape)
print(data[1].shape)
print(data[0])