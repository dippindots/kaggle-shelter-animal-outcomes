'''
Created on Apr 29, 2016

Based on [Classifier comparison]
(http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html).

@author: Paul Reiners
'''
import time

from core.evaluation import log_loss
from core.learning.classifiers.ada_boost_predictor import AdaBoostPredictor
from core.learning.classifiers.decision_tree_predictor \
    import DecisionTreePredictor
from core.learning.classifiers.linear_descriminant_analysis_predictor \
    import LinearDiscriminantAnalysisPredictor
from core.learning.classifiers.naive_bayes_predictor import NaiveBayesPredictor
from core.learning.classifiers.nearest_neighbors_predictor \
    import NearestNeighborsPredictor
from core.learning.classifiers.quadratic_descriminant_analysis_predictor \
    import QuadraticDiscriminantAnalysisPredictor
from core.learning.classifiers.random_forest_predictor \
    import RandomForestPredictor
from core.preprocessing.feature_extraction_scaling import get_data
from core.util import split_data, preprocess_data


if __name__ == '__main__':
    names = ["Nearest Neighbors", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes",
             "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"]
    # Slow: "Linear SVM", "RBF SVM"
    possible_outcomes = [
        'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    # Iterate over AnimalType
    for animal_type in ['Cat', 'Dog']:
        classifiers = [
            NearestNeighborsPredictor(animal_type),
            DecisionTreePredictor(animal_type),
            RandomForestPredictor(animal_type),
            AdaBoostPredictor(),
            NaiveBayesPredictor(),
            LinearDiscriminantAnalysisPredictor(animal_type),
            QuadraticDiscriminantAnalysisPredictor()]
        # Slow: LinearSVMPredictor(), RBF_SVMPredictor()

        train_data = get_data('../data/train.csv')
        print animal_type
        train_data = train_data[train_data['AnimalType'] == animal_type]
        train_data = train_data.drop(['AnimalType'], axis=1)
        train_data = preprocess_data(train_data, animal_type)
        train_data = train_data.dropna()
        X_train, y_train, X_test, y_test = split_data(train_data)

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            clf_X_train = X_train
            clf_X_test = X_test

            print "\t{} {}".format(name, time.ctime())
            clf.fit(clf_X_train, y_train)
            predictions_df = clf.predict(clf_X_test)
            ll = log_loss(
                y_test, 'OutcomeType', predictions_df, possible_outcomes)

            print "\t\tscore: %.5f" % ll
