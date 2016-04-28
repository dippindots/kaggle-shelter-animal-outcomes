'''
Created on Apr 27, 2016

MyLLScore:     3.70919
KaggleLLScore: 3.59477

@author: Paul Reiners
'''
from sklearn.neighbors import KNeighborsClassifier

from predictor_base import PredictorBase


class NearestNeighborsPredictor(PredictorBase):
    '''
    Uses k-nearest neighbors.
    '''

    def __init__(self):
        self.clf = KNeighborsClassifier(7)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df
