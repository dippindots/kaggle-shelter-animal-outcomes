'''
Created on Apr 27, 2016

MyLLScore:     1.24175
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn.ensemble import RandomForestClassifier
from predictor_base import PredictorBase


class RandomForestPredictor(PredictorBase):
    '''
    Random Forest
    '''

    def __init__(self):
        self.clf = RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df
