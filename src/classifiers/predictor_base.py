'''
Created on Apr 27, 2016

@author: Paul Reiners
'''
import abc
import pandas as pd


class PredictorBase(object):
    '''
    Abstract base predictor class
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, X_train, y_train):
        """Fit model."""
        return

    @abc.abstractmethod
    def predict(self, X_test):
        """Make predictions."""
        return

    def find_best_params(self):
        """Use grid search to find best params"""
        pass

    def bundle_predictions(self, predictions):
        n = len(predictions)
        predictions = predictions.transpose()
        predictions_data = {
            'ID': range(1, n + 1), 'Adoption': predictions[0],
            'Died': predictions[1], 'Euthanasia': predictions[2],
            'Return_to_owner': predictions[3], 'Transfer': predictions[4]}
        predictions_df = pd.DataFrame(predictions_data)

        return predictions_df
