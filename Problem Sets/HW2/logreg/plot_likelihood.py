import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt 

data1 = pd.read_table("../data1.txt", header = None, sep = ",");
data1 = data1.as_matrix()
data2 = pd.read_table("../data2.txt", header = None, sep = ",");
data2 = data2.as_matrix()
plt.figure()
plt.plot(data1[:,0],data1[:,1], '-b', label='feature values')
plt.plot(data2[:,0],data2[:,1], '-r', label='tf-idf values')
plt.legend(loc='lower right')
plt.xlabel('Iteration')
plt.ylabel('Holdout Accuracy')
plt.title('Feature values vs tf-idf')
plt.show()
    