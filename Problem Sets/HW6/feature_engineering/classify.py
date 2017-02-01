from __future__ import division
from csv import DictReader, DictWriter
import numpy as np
from numpy import array
from stemming.porter2 import stem
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
TEST = True


def removepunc(text):
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    text = ' '.join([stem(word) for word in text.split()])
    return text


class Featurizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', preprocessor=removepunc)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))

    temp = (x[kTEXT_FIELD] + ' ' + x['page'] + ' ' + re.sub(r"(?<=\w)([A-Z])", r" \1", x['trope']) for x in train)
    X = feat.train_feature(temp)
    num_present = []
    for val in temp:
        num_present.append(bool(re.search(r'\d', val)))
    np.hstack((X, num_present))

    y = array(list(labels.index(x[kTARGET_FIELD]) for x in train))
    print(len(train), len(y))
    print(set(y))

    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    if TEST:
        lr.fit(X, y)
        temp = (x[kTEXT_FIELD] + ' ' + x['page'] + ' ' + re.sub(r"(?<=\w)([A-Z])", r" \1", x['trope']) for x in test)
        x_test = feat.test_feature(temp)
        num_present = []
        for val in temp:
            num_present.append(bool(re.search(r'\d', val)))
        np.hstack((x_test, num_present))
        predictions = lr.predict(x_test)
        o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
        o.writeheader()
        for ii, pp in zip([x['id'] for x in test], predictions):
            d = {'id': ii, 'spoiler': labels[pp]}
            o.writerow(d)
    else:
        kf = KFold(len(train), n_folds=5, shuffle=True)
        t_acc = v_acc = 0
        for train_index, val_index in kf:
            x_train, x_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            lr.fit(x_train, y_train)

            t_acc += accuracy_score(y_train, lr.predict(x_train))
            v_acc += accuracy_score(y_val, lr.predict(x_val))

        print "Logistic Regression training accuracy: ", t_acc/5
        print "Logistic Regression validation accuracy: ", v_acc/5
