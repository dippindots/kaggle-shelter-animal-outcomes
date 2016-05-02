'''
Created on Apr 27, 2016

MyLLScore:     31.89242
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn.naive_bayes import GaussianNB

from core.learning.classifiers.predictor_base import PredictorBase


class NaiveBayesPredictor(PredictorBase):
    '''
    Naive Bayes
    '''

    def __init__(self):
        self.clf = GaussianNB()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df
