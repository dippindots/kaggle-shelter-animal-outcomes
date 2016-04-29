'''
Created on Apr 27, 2016

MyLLScore:     0.99518
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from predictor_base import PredictorBase


class LinearDiscriminantAnalysisPredictor(PredictorBase):
    '''
    Linear Discriminant Analysis
    '''

    def __init__(self):
        self.clf = LinearDiscriminantAnalysis()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df
