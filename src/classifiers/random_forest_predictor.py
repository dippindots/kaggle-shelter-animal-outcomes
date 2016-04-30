'''
Created on Apr 27, 2016

MyLLScore:     0.82221
KaggleLLScore: 0.84454

@author: Paul Reiners
'''
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from classifiers.predictor_base import PredictorBase
from util import get_data, preprocess_data


class RandomForestPredictor(PredictorBase):
    '''
    Random Forest
    '''

    def __init__(self):
        # n_estimators
        # 320: 0.85407
        self.clf = RandomForestClassifier(
            max_depth=8, n_estimators=320, max_features=4)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df

    def find_best_params(self):
        parameters = {
            'n_estimators': [2, 5, 10, 20, 40, 80, 160, 320, 640, 1280],
            'max_depth': [4, 8, 16],
            'max_features': [1, 2, 4, 8]}
        rf = RandomForestClassifier()
        clf = grid_search.GridSearchCV(rf, parameters)
        train_data = get_data('../../data/train.csv')
        train_data = preprocess_data(train_data)
        train_data = train_data.dropna()
        X = train_data.drop(['OutcomeType'], axis=1)
        y = train_data['OutcomeType']
        k_best = SelectKBest(chi2, k=10)
        X = k_best.fit_transform(X, y)
        clf.fit(X, y)
        print clf.best_params_

if __name__ == '__main__':
    predictor = RandomForestPredictor()
    predictor.find_best_params()
