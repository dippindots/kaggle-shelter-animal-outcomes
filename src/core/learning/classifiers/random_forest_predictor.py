'''
Created on Apr 27, 2016

Thin wrapper around RandomForestClassifier.

MyLLScore:     0.79955
KaggleLLScore: 0.79167

@author: Paul Reiners
'''
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier

from core.learning.classifiers.predictor_base import PredictorBase
from core.preprocessing.feature_extraction_scaling import get_data
from core.preprocessing.feature_selection import select_features


class RandomForestPredictor(PredictorBase):
    '''
    Random Forest
    '''

    def __init__(self, animal_type):
        """ Initialize class instance with type of animal. """
        self.animal_type = animal_type
        self.base_args = {'random_state': 1}
        args = self.base_args.copy()
        if self.animal_type == "Cat":
            args.update(
                {'n_estimators': 80, 'criterion': 'gini', 'max_depth': 10})
        elif self.animal_type == "Dog":
            args.update(
                {'n_estimators': 80, 'criterion': 'entropy', 'max_depth': 8})
        else:
            raise RuntimeError("Incorrect animal type")

        self.clf = RandomForestClassifier(**args)

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
            'n_estimators': [40, 80, 160, 320],
            'max_depth': range(7, 13), 'criterion': ['entropy', 'gini']}
        rf = RandomForestClassifier(self.base_args)
        clf = grid_search.GridSearchCV(rf, parameters)
        train_data = get_data('../data/train.csv')
        if 'SexuponOutcome' in train_data.columns:
            train_data = train_data[train_data.SexuponOutcome.notnull()]
        train_data = train_data[train_data.AgeuponOutcome.notnull()]
        train_data = select_features(train_data, self.animal_type)
        X = train_data.drop(['OutcomeType'], axis=1)
        y = train_data['OutcomeType']
        clf.fit(X, y)
        print clf.best_params_

if __name__ == '__main__':
    print 'Cat'
    predictor = RandomForestPredictor('Cat')
    predictor.find_best_params()
    print 'Dog'
    predictor = RandomForestPredictor('Dog')
    predictor.find_best_params()
