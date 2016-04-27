'''
Created on Apr 27, 2016

@author: Paul Reiners
'''
import abc

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
        