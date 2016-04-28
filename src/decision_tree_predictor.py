'''
Created on Apr 27, 2016

MyLLScore:     0.94061
KaggleLLScore: 0.90950

@author: Paul Reiners
'''
from sklearn import tree

from predictor_base import PredictorBase


class DecisionTreePredictor(PredictorBase):
    '''
    Uses decision tree.
    '''

    def __init__(self):
        self.clf = tree.DecisionTreeClassifier(max_depth=6)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df
