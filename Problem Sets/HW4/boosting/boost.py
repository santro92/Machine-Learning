import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone
import matplotlib.pyplot as plt
import math
from sklearn import metrics

np.random.seed(1234)

class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set
        train_set, valid_set, test_set = cPickle.load(f)

        # Extract only 4's and 9's for training set
        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]
        self.y_train = np.array([1 if y == 9 else -1 for y in self.y_train])

        # Shuffle the training data
        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 4's and 9's for validation set
        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]
        self.y_valid = np.array([1 if y == 9 else -1 for y in self.y_valid])

        # Extract only 4's and 9's for test set
        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]
        self.y_test = np.array([1 if y == 9 else -1 for y in self.y_test])

        f.close()

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1)):
        """
        Create a new adaboost classifier.

        Args:
            n_learners (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner

        Attributes:
            base (estimator): Your general weak learner
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners.
            learners (list): List of weak learner instances.
        """

        self.n_learners = n_learners
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []

    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners.

        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data
            y_train (ndarray): [n_samples] ndarray of data
        """

        # TODO

        # Hint: You can create and train a new instantiation
        # of your sklearn weak learner as follows

        w = np.ones(len(y_train))
        w = w/sum(w)
        for k in range(self.n_learners):
        	h = clone(self.base)
        	h.fit(X_train, y_train, sample_weight=w)
        	self.learners.append(h)
        	hx = h.predict(X_train)
        	w_sum = 0
        	for i in range(len(y_train)):
        		if(hx[i] != y_train[i]):
        			w_sum += w[i]
        	err = float(w_sum)/sum(w)
        	self.alpha[k] = 0.5 * math.log((1-err)/err)
        	e_vals = -1*self.alpha[k]*hx*y_train
        	w = w * np.exp(e_vals)
        	w = w/sum(w)

    def predict(self, X):
        """
        Adaboost prediction for new data X.

        Args:
            X (ndarray): [n_samples x n_features] ndarray of data

        Returns:
            [n_samples] ndarray of predicted labels {-1,1}
        """

        # TODO
    	sum = 0
    	for k in range(self.n_learners):
    		h = self.learners[k]
    		sum += self.alpha[k]*(h.predict(X))
        return np.sign(sum)

    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.

        Args:
            X (ndarray): [n_samples x n_features] ndarray of data
            y (ndarray): [n_samples] ndarray of true labels

        Returns:
            Prediction accuracy (between 0.0 and 1.0).
        """

        # TODO
        lsum = 0
        pred = self.predict(X)
        for i in range(len(y)):
        	if(pred[i] == y[i]):
        		lsum += 1
        return float(lsum)/len(y)

    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting
        for monitoring purposes, such as to determine the score on a
        test set after each boost.

        Args:
            X (ndarray): [n_samples x n_features] ndarray of data
            y (ndarray): [n_samples] ndarray of true labels

        Returns:
            [n_learners] ndarray of scores
        """

        # TODO
        learners_score = np.zeros(self.n_learners)
        for l in range(1,self.n_learners+1):
            labels = 0
            for k in range(l):
                h = self.learners[k]
                labels += self.alpha[k]*(h.predict(X))
            labels = np.sign(labels)

            lsum = 0
            for j in range(len(y)):
                if(labels[j] == y[j]):
                    lsum += 1
            learners_score[l-1] = float(lsum)/len(y)
        return learners_score

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

	parser = argparse.ArgumentParser(description='AdaBoost classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	parser.add_argument('--n_learners', type=int, default=50,
                        help="Number of weak learners to use in boosting")
	args = parser.parse_args()

	data = FoursAndNines("../data/mnist.pkl.gz")

    # An example of how your classifier might be called

	for depth in [5,6,8,10,12,15]:
		clf = AdaBoost(n_learners=200, base=Perceptron(n_iter=depth))#DecisionTreeClassifier(max_depth=depth, criterion="entropy"))
        clf.fit(data.x_train, data.y_train)
    	trainscore = clf.staged_score(data.x_train, data.y_train)
        validscore = clf.staged_score(data.x_valid, data.y_valid)
        file = open("/home/santa/Dropbox/ML/HW4/boosting/train"+str(depth)+".txt", "a")
        for i in range(len(trainscore)):
        	file.write(str(1-trainscore[i]) + "\n")
        file.close()
        file = open("/home/santa/Dropbox/ML/HW4/boosting/valid"+str(depth)+".txt", "a")
        for i in range(len(validscore)):
        	file.write(str(1-validscore[i]) + "\n")
        file.close()
