'''
Created on Apr 27, 2016

TAKES MORE THAN A HALF-HOUR TO RUN.

MyLLScore:
KaggleLLScore:

@author: Paul Reiners
'''
import time

from sklearn import grid_search
from sklearn.svm import SVC

from core.learning.classifiers.predictor_base import PredictorBase
from core.preprocessing.feature_extraction_scaling import get_data
from core.preprocessing.feature_selection import select_features


class RBF_SVMPredictor(PredictorBase):
    '''
    RBF SVM
    '''

    def __init__(self, animal_type):
        self.animal_type = animal_type
        self.clf = SVC(
            kernel='rbf', random_state=0, gamma=0.1, C=10.0, probability=True)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df

    def find_best_params(self):
        parameters = {'C': [1.0, 10.0, 100.0], 'gamma': [0.01, 0.10, 1.0]}
        svc = SVC(kernel='rbf')
        clf = grid_search.GridSearchCV(svc, parameters)
        train_data = get_data('../data/train.csv')
        train_data = select_features(train_data, self.animal_type)
        X = train_data.drop(['OutcomeType'], axis=1)
        y = train_data['OutcomeType']
        clf.fit(X, y)
        print clf.best_params_

if __name__ == '__main__':
    print "{} {}".format('Cat', time.ctime())
    predictor = RBF_SVMPredictor('Cat')
    predictor.find_best_params()
    print "{} {}".format('Dog', time.ctime())
    predictor = RBF_SVMPredictor('Dog')
    predictor.find_best_params()
