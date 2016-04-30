'''
Created on Apr 27, 2016

MyLLScore:     26.94625
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from classifiers.predictor_base import PredictorBase


class QuadraticDiscriminantAnalysisPredictor(PredictorBase):
    '''
    Quadratic Discriminant Analysis
    '''

    def __init__(self):
        self.clf = QuadraticDiscriminantAnalysis()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df

    def get_k_best_k(self):
        return 4
