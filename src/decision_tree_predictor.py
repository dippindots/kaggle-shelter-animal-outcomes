'''
Created on Apr 27, 2016

MyLLScore:     10.69225
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn import tree

import pandas as pd
from predictor_base import PredictorBase


class DecisionTreePredictor(PredictorBase):
    '''
    Uses decision tree.
    '''

    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions = predictions.transpose()
        n = len(X_test)
        predictions_data = {
            'ID': range(1, n + 1), 'Adoption': predictions[0],
            'Died': predictions[1], 'Euthanasia': predictions[2],
            'Return_to_owner': predictions[3], 'Transfer': predictions[4]}
        predictions_df = pd.DataFrame(predictions_data)

        return predictions_df
