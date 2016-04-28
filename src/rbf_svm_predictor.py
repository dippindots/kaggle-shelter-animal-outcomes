'''
Created on Apr 27, 2016

TAKES MORE THAN A HALF-HOUR TO RUN.

MyLLScore:
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn.svm import SVC

from predictor_base import PredictorBase


class RBF_SVMPredictor(PredictorBase):
    '''
    RBF SVM
    '''

    def __init__(self):
        self.clf = SVC(gamma=2, C=1, probability=True)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df
