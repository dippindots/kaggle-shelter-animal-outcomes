'''
Created on Apr 27, 2016

MyLLScore:     1.00052
KaggleLLScore: 1.00036

@author: Paul Reiners
'''
from sklearn import grid_search
from sklearn.neighbors import KNeighborsClassifier

from core.learning.classifiers.predictor_base import PredictorBase
from core.preprocessing.feature_extraction_scaling import get_data
from core.util import preprocess_data


class NearestNeighborsPredictor(PredictorBase):
    '''
    Uses k-nearest neighbors.
    '''

    def __init__(self, animal_type):
        self.animal_type = animal_type
        if self.animal_type == "Cat":
            args = {'n_neighbors': 20}
        elif self.animal_type == "Dog":
            args = {'n_neighbors': 40}
        else:
            raise RuntimeError("Incorrect animal type")
        self.clf = KNeighborsClassifier(**args)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df

    def find_best_params(self):
        parameters = {'n_neighbors': [10, 20, 40, 60]}
        knn = KNeighborsClassifier()
        clf = grid_search.GridSearchCV(knn, parameters)
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
    print 'Cat'
    predictor = NearestNeighborsPredictor('Cat')
    predictor.find_best_params()
    print 'Dog'
    predictor = NearestNeighborsPredictor('Dog')
    predictor.find_best_params()
