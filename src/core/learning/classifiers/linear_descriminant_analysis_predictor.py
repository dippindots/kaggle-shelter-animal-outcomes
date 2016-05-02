'''
Created on Apr 27, 2016

MyLLScore:     0.99518
KaggleLLScore:

@author: Paul Reiners
'''
from sklearn import grid_search
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from core.learning.classifiers.predictor_base import PredictorBase
from core.preprocessing.feature_extraction_scaling import get_data
from core.preprocessing.feature_selection import select_features


class LinearDiscriminantAnalysisPredictor(PredictorBase):
    '''
    Linear Discriminant Analysis
    '''

    def __init__(self, animal_type):
        self.animal_type = animal_type
        self.clf = LinearDiscriminantAnalysis()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.clf.predict_proba(X_test)
        predictions_df = self.bundle_predictions(predictions)

        return predictions_df

    def find_best_params(self):
        parameters = {'solver': ['svd', 'lsqr', 'eigen']}
        knn = LinearDiscriminantAnalysis()
        clf = grid_search.GridSearchCV(knn, parameters)
        train_data = get_data('../data/train.csv')
        train_data = select_features(train_data, self.animal_type)
        X = train_data.drop(['OutcomeType'], axis=1)
        y = train_data['OutcomeType']
        clf.fit(X, y)
        print clf.best_params_

if __name__ == '__main__':
    print 'Cat'
    predictor = LinearDiscriminantAnalysisPredictor('Cat')
    predictor.find_best_params()
    print 'Dog'
    predictor = LinearDiscriminantAnalysisPredictor('Dog')
    predictor.find_best_params()
