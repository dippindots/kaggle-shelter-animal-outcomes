'''
Created on May 2, 2016

Functions for data sampling.

@author: Paul Reiners
'''
from sklearn.cross_validation import train_test_split


def split_data(train_data):
    """ Split data into training and test sets. """
    X = train_data.drop(['OutcomeType'], axis=1)
    y = train_data['OutcomeType']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    return X_train, y_train, X_test, y_test
