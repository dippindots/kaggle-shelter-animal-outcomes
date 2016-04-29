'''
Created on Apr 29, 2016

Based on [Classifier comparison]
(http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html).

@author: Paul Reiners
'''
import time

from classifiers.ada_boost_predictor import AdaBoostPredictor
from classifiers.decision_tree_predictor import DecisionTreePredictor
from classifiers.linear_descriminant_analysis_predictor \
    import LinearDiscriminantAnalysisPredictor
from classifiers.naive_bayes_predictor import NaiveBayesPredictor
from classifiers.nearest_neighbors_predictor import NearestNeighborsPredictor
from classifiers.quadratic_descriminant_analysis_predictor \
    import QuadraticDiscriminantAnalysisPredictor
from classifiers.random_forest_predictor import RandomForestPredictor
from util import get_data, split_data, measure_log_loss_of_predictor, \
    clean_data


if __name__ == '__main__':
    names = ["Nearest Neighbors", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes",
             "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"]
    # Slow: "Linear SVM", "RBF SVM"
    classifiers = [
        NearestNeighborsPredictor(),
        DecisionTreePredictor(),
        RandomForestPredictor(),
        AdaBoostPredictor(),
        NaiveBayesPredictor(),
        LinearDiscriminantAnalysisPredictor(),
        QuadraticDiscriminantAnalysisPredictor()]
    # Slow: LinearSVMPredictor(), RBF_SVMPredictor()
    train_data = get_data('../data/train.csv')
    train_data = clean_data(train_data)
    train_data = train_data.dropna()
    X_train, y_train, X_test, y_test = split_data(train_data)

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print name, time.ctime()
        ll = measure_log_loss_of_predictor(
            X_train, y_train, X_test, y_test, clf)
        print "\tscore: %.5f\n" % ll
