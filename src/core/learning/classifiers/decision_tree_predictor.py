'''
Created on Apr 27, 2016

MyLLScore:     0.92471
KaggleLLScore: 0.90939

@author: Paul Reiners
'''
from sklearn import grid_search
from sklearn import tree

from core.learning.classifiers.predictor_base import PredictorBase
from core.preprocessing.feature_extraction_scaling import get_data
from core.util import preprocess_data


class DecisionTreePredictor(PredictorBase):
    '''
    Uses decision tree.
    '''

    def __init__(self, animal_type):
        self.animal_type = animal_type
        args = {'criterion': 'entropy', 'max_depth': 6}
        self.clf = tree.DecisionTreeClassifier(**args)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df

    def find_best_params(self):
        parameters = {
            'criterion': ('gini', 'entropy'), 'max_depth': [3, 6, 12]}
        decision_tree = tree.DecisionTreeClassifier()
        clf = grid_search.GridSearchCV(decision_tree, parameters)
        train_data = get_data('../data/train.csv')
        train_data = train_data[train_data['AnimalType'] == self.animal_type]
        train_data = train_data.drop(['AnimalType'], axis=1)
        train_data = preprocess_data(train_data, self.animal_type)
        train_data = train_data.dropna()
        X = train_data.drop(['OutcomeType'], axis=1)
        y = train_data['OutcomeType']
        clf.fit(X, y)
        print clf.best_params_

if __name__ == '__main__':
    print 'Dog'
    decision_tree_predictor = DecisionTreePredictor('Dog')
    decision_tree_predictor.find_best_params()
    print 'Cat'
    decision_tree_predictor = DecisionTreePredictor('Cat')
    decision_tree_predictor.find_best_params()
