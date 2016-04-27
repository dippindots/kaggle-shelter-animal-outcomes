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
    def predict(self, X_train, X_test, y_train):
        """Make predictions."""
        return
        