'''
Created on Apr 27, 2016

Thin wrapper around XGBClassifier.

MyLLScore:
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn import grid_search

from core.learning.classifiers.predictor_base import PredictorBase
from core.preprocessing.feature_extraction_scaling import get_data
from core.preprocessing.feature_selection import select_features
import xgboost as xgb


class XGBPredictor(PredictorBase):
    '''
    XGB
    '''

    def __init__(self, animal_type, is_adult):
        """ Initialize class instance with type of animal. """
        self.animal_type = animal_type
        self.is_adult = is_adult
        self.base_args = {'objective': 'multi:softprob'}
        args = self.base_args.copy()
        if self.animal_type == "Cat":
            if self.is_adult == 0:
                updated_args = {
                    'n_estimators': 1200, 'learning_rate': 0.05,
                    'max_depth': 2}
            else:
                updated_args = {
                    'n_estimators': 38, 'learning_rate': 0.2, 'max_depth': 2}
        elif self.animal_type == "Dog":
            if self.is_adult == 0:
                updated_args = {
                    'n_estimators': 75, 'learning_rate': 0.025, 'max_depth': 4}
            else:
                updated_args = {
                    'n_estimators': 75, 'learning_rate': 0.025, 'max_depth': 4}
        else:
            raise RuntimeError("Incorrect animal type")

        args.update(updated_args)

        self.clf = xgb.XGBClassifier(**args)

    def fit(self, X_train, y_train):
        """ Fit the random forest model. """
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        """ Make prediction. """
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df

    def find_best_params(self):
        """ Find best hyper-parameters for the classifier. """
        parameters = {
            'n_estimators': [38, 75, 150, 300, 600, 1200],
            'max_depth': range(2, 6),
            'learning_rate': [0.025, 0.050, 0.100, 0.200]}
        rf = xgb.XGBClassifier(self.base_args)
        clf = grid_search.GridSearchCV(rf, parameters)
        train_data = get_data('../data/train.csv')
        if 'SexuponOutcome' in train_data.columns:
            train_data = train_data[train_data.SexuponOutcome.notnull()]
        train_data = train_data[train_data.AgeuponOutcome.notnull()]
        train_data = select_features(
            train_data, self.animal_type, self.is_adult)
        X = train_data.drop(['OutcomeType'], axis=1)
        y = train_data['OutcomeType']
        clf.fit(X, y)
        print clf.best_params_

if __name__ == '__main__':
    for animal_type in ['Cat', 'Dog']:
        print animal_type
        for is_adult in [0, 1]:
            print '\t' + str(is_adult)
            predictor = XGBPredictor(animal_type, is_adult)
            predictor.find_best_params()
