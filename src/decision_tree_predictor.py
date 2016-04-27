'''
Created on Apr 27, 2016

MyLLScore:      
KaggleLLScore: 

@author: Paul Reiners
'''
from predictor_base import PredictorBase
from sklearn import tree

class DecisionTreePredictor(PredictorBase):
    '''
    Uses decision tree.
    '''


    def predict(self, X_train, X_test, y_train):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        print clf
        
