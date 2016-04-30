'''
Created on Apr 27, 2016

MyLLScore:     0.88390
KaggleLLScore: 0.86738

@author: Paul Reiners
'''
from sklearn.ensemble import RandomForestClassifier
from classifiers.predictor_base import PredictorBase


class RandomForestPredictor(PredictorBase):
    '''
    Random Forest
    '''

    def __init__(self):
        self.clf = RandomForestClassifier(
            max_depth=8, n_estimators=10, max_features=1)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df
