'''
Created on Apr 27, 2016

MyLLScore:     0.82221
KaggleLLScore: 0.84454

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
        self.animal_type = animal_type
        if self.animal_type == "Cat":
            args = {'n_estimators': 40, 'max_depth': 8}
        elif self.animal_type == "Dog":
            args = {'n_estimators': 40, 'max_depth': 7}
        else:
            raise RuntimeError("Incorrect animal type")

        self.clf = RandomForestClassifier(**args)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df

    def find_best_params(self):
        parameters = {
            'n_estimators': [40, 80, 160, 320, 640],
            'max_depth': range(5, 10)}
        rf = RandomForestClassifier()
        clf = grid_search.GridSearchCV(rf, parameters)
        train_data = get_data('../data/train.csv')
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
