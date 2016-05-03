'''
Created on Apr 27, 2016

MyLLScore:
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn.svm import SVC

from core.learning.classifiers.predictor_base import PredictorBase


class LinearSVMPredictor(PredictorBase):
    '''
    Linear SVM
    '''

    def __init__(self):
        self.clf = SVC(kernel="linear", C=0.025, probability=True)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df
