'''
Created on Apr 27, 2016

MyLLScore:     20.61577
KaggleLLScore: 20.25113

@author: Paul Reiners
'''
import pandas as pd
from predictor_base import PredictorBase


class BaseLinePredictor(PredictorBase):
    '''
    All adopted benchmark
    '''

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        test_n = len(X_test)
        zeroes = [0] * test_n
        predictions_data = {
            'ID': test_n, 'Adoption': [1] * test_n, 'Died': zeroes,
            'Euthanasia': zeroes, 'Return_to_owner': zeroes,
            'Transfer': zeroes}
        predictions_df = pd.DataFrame(predictions_data)

        return predictions_df
