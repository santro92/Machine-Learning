import argparse
import numpy as np 

from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC
from sklearn import metrics

class ThreesAndEights:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.
		
		import cPickle, gzip

        # Load the dataset
		f = gzip.open(location, 'rb')

		train_set, valid_set, test_set = cPickle.load(f)

		self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0],:]
		self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0]]

		shuff = np.arange(self.x_train.shape[0])
		np.random.shuffle(shuff)
		self.x_train = self.x_train[shuff,:]
		self.y_train = self.y_train[shuff]

		self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0],:]
		self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0]]
		
		self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0],:]
		self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0]]

		f.close()

def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='SVM classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	args = parser.parse_args()

	data = ThreesAndEights("../data/mnist.pkl.gz")

	sv = SVC(kernel='linear')
	sv.fit(data.x_train,data.y_train)
	# for kernel_name in ['linear','rbf']:
		# for c in [0.01,0.1,1,2,5,10,100]:
			# sv = SVC(C=c, kernel=kernel_name)
	
			# print str(c) + "\t" + kernel_name + "\t" + str(sv.score(data.x_valid, data.y_valid))	
	
	# print(metrics.confusion_matrix(data.y_valid, sv.predict(data.x_valid)))
	# summarize the fit of the model
	# print(metrics.classification_report(expected, predicted))
	

	# -----------------------------------
	# Plotting Examples 
	# -----------------------------------

	# Display in on screen  
	mnist_digit_show(data.x_train[ sv.support_[0],:],"3a.png")
	mnist_digit_show(data.x_train[ sv.support_[1],:],"3b.png")
	mnist_digit_show(data.x_train[ sv.support_[sv.n_support_[0]],:],"8a.png")
	mnist_digit_show(data.x_train[ sv.support_[sv.n_support_[0]+1],:],"8b.png")
	# Plot image to file 
	# mnist_digit_show(data.x_train[1,:], "mnistfig.png")