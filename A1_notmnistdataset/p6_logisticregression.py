from  A1_notmnistdataset.p5_findduplication import DataExploration
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from time import time
from sklearn.cross_validation import StratifiedShuffleSplit


class P6_LRModel(DataExploration):
    def __init__(self):
        DataExploration.__init__(self)
        self.reshapeData()
        return
    def reshapeData(self):
        #reshape the data so that its shape changes to two dimension
        self.train_dataset = self.train_dataset.reshape(self.train_dataset.shape[0], -1)
        self.test_dataset = self.test_dataset.reshape(self.test_dataset.shape[0], -1)
        self.valid_dataset = self.valid_dataset.reshape(self.valid_dataset.shape[0], -1)
        return
    def getTrainSet(self, num):
        if num == -1:
            return self.train_dataset, self.train_labels
        train_dataset = self.train_dataset[:num]
        train_labels = self.train_labels[:num]
        return train_dataset, train_labels
    def runOutofbox(self):
        print("run out of the box logistic regression classifier")
        training_size = [50,100,1000,5000,-1]
        for size in training_size:
            t0 = time()
            train_dataset, train_labels = self.getTrainSet(size)
            print("training set size: {}".format(train_labels.shape[0]))
            logistic = linear_model.LogisticRegression()
            estimator = logistic
            estimator.fit(train_dataset, train_labels)
            y_pred = estimator.predict(self.test_dataset)
            accuracy = accuracy_score(self.test_labels, y_pred)
#             mean_accuracy = estimator.score(self.test_dataset, self.test_labels)
            print("training and prediction time:{}s".format(round(time()-t0, 3)))
            print("accuracy: {}".format(accuracy))
        return
    def getTunedParamterOptions(self):
        params = {'penalty': ['l2'], 'C': np.logspace(-3, 3, 7), 'solver': ['lbfgs'], 'multi_class': ['multinomial']}
        return params
    def runGridSearch(self):
        print("run grid search")
        training_size = [100,1000,5000,-1]
        for size in training_size:
            t0 = time()
            train_dataset, train_labels = self.getTrainSet(size)
            print("training set size: {}".format(train_labels.shape[0]))
            estimator = linear_model.LogisticRegression()
            estimator = GridSearchCV(estimator, self.getTunedParamterOptions(), 
                                     cv=StratifiedShuffleSplit(train_labels, n_iter=10,random_state = 42),scoring= 'accuracy')
            estimator.fit(train_dataset, train_labels)
            print("best parameters: {}".format(estimator.best_params_))
            print("best scores: {}".format(estimator.best_score_))
            
            y_pred = estimator.best_estimator_.predict(self.test_dataset)
            accuracy = accuracy_score(self.test_labels, y_pred)
#             mean_accuracy = estimator.score(self.test_dataset, self.test_labels)
            print("training and prediction time:{}s".format(round(time()-t0, 3)))
            print("accuracy: {}".format(accuracy))
        return
    def run(self):
#         self.runOutofbox()
        self.runGridSearch()
        return
    


if __name__ == "__main__":   
    obj= P6_LRModel()
    obj.run()