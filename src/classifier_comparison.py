'''
Created on Apr 29, 2016

Based on [Classifier comparison]
(http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html).

@author: Paul Reiners
'''
from classifiers.ada_boost_predictor import AdaBoostPredictor
from classifiers.decision_tree_predictor import DecisionTreePredictor
from classifiers.linear_descriminant_analysis_predictor \
    import LinearDiscriminantAnalysisPredictor
from classifiers.linear_svm_predictor import LinearSVMPredictor
from classifiers.naive_bayes_predictor import NaiveBayesPredictor
from classifiers.nearest_neighbors_predictor import NearestNeighborsPredictor
from classifiers.quadratic_descriminant_analysis_predictor \
    import QuadraticDiscriminantAnalysisPredictor
from classifiers.random_forest_predictor import RandomForestPredictor
from classifiers.rbf_svm_predictor import RBF_SVMPredictor
from util import get_data, split_data, measure_log_loss_of_predictor


if __name__ == '__main__':
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes",
             "Linear Discriminant Analysis", "Quadratic Discriminant Analysis"]
    classifiers = [
        NearestNeighborsPredictor(),
        LinearSVMPredictor(),
        RBF_SVMPredictor(),
        DecisionTreePredictor(),
        RandomForestPredictor(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostPredictor(),
        NaiveBayesPredictor(),
        LinearDiscriminantAnalysisPredictor(),
        QuadraticDiscriminantAnalysisPredictor()]
    train_data = get_data('../data/train.csv', 'train')
    train_data = train_data.dropna()

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        X_train, y_train, X_test, y_test = split_data(train_data)

        ll = measure_log_loss_of_predictor(
            X_train, y_train, X_test, y_test, clf)
        print name
        print "\tscore: %.5f\n" % ll
