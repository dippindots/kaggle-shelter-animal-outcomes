'''
Created on Apr 27, 2016

@author: Paul Reiners
'''
import abc

from core.learning.performance_metrics import bundle_predictions


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
        return bundle_predictions(predictions)

    def get_k_best_k(self):
        return 10
