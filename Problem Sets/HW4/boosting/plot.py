import matplotlib.pyplot as plt
import numpy as np

fileList = ["12","15"]
for file in fileList:
    filename = "/home/santa/Dropbox/ML/HW4/boosting/train"+file+".txt"
    lines = [line.rstrip('\n') for line in open(filename)]
    plt.plot(np.array(lines), label='Train Error')
    filename = "/home/santa/Dropbox/ML/HW4/boosting/valid"+file+".txt"
    lines = [line.rstrip('\n') for line in open(filename)]
    plt.plot(np.array(lines), label='Test Error')
    plt.xlabel('boosting iteration')
    plt.ylabel('error')
    plt.title('Iteration = ' + file)
    plt.legend(loc='upper right', fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig("/home/santa/Dropbox/ML/HW4/boosting/"+file+".png")
    plt.clf()
