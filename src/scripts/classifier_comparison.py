'''
Created on Apr 29, 2016

Based on [Classifier comparison]
(http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html).

Measures the performance of various types of classifiers.

@author: Paul Reiners
'''
from core.learning.classifiers.ada_boost_predictor import AdaBoostPredictor
from core.learning.classifiers.decision_tree_predictor \
    import DecisionTreePredictor
from core.learning.classifiers.linear_descriminant_analysis_predictor \
    import LinearDiscriminantAnalysisPredictor
from core.learning.classifiers.linear_svm_predictor import LinearSVMPredictor
from core.learning.classifiers.naive_bayes_predictor import NaiveBayesPredictor
from core.learning.classifiers.nearest_neighbors_predictor \
    import NearestNeighborsPredictor
from core.learning.classifiers.quadratic_descriminant_analysis_predictor \
    import QuadraticDiscriminantAnalysisPredictor
from core.learning.classifiers.random_forest_predictor \
    import RandomForestPredictor
from core.learning.classifiers.rbf_svm_predictor import RBF_SVMPredictor
from core.learning.classifiers.xgb_predictor import XGBPredictor
from core.learning.performance_metrics import log_loss
from core.preprocessing.feature_extraction_scaling import get_data
from core.preprocessing.feature_selection import select_features
from core.preprocessing.sampling import split_data


if __name__ == '__main__':
    names = ["Nearest Neighbors", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes",
             "Linear Discriminant Analysis", "Quadratic Discriminant Analysis",
             "Linear SVM", "RBF SVM", "XGBoost"]
    possible_outcomes = [
        'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    # Iterate over AnimalType
    for animal_type in ['Cat', 'Dog']:
        print "*", animal_type
        for is_adult in [0, 1]:
            print " -", is_adult
            classifiers = [
                NearestNeighborsPredictor(animal_type),
                DecisionTreePredictor(animal_type),
                RandomForestPredictor(animal_type),
                AdaBoostPredictor(),
                NaiveBayesPredictor(),
                LinearDiscriminantAnalysisPredictor(animal_type),
                QuadraticDiscriminantAnalysisPredictor(),
                LinearSVMPredictor(animal_type), RBF_SVMPredictor(animal_type),
                XGBPredictor(animal_type, is_adult)]

            train_data = get_data('../data/train.csv')
            train_data = train_data[train_data.SexuponOutcome.notnull()]
            train_data = train_data[train_data.AgeuponOutcome.notnull()]

            train_data = select_features(train_data, animal_type, is_adult)
            X_train, y_train, X_test, y_test = split_data(train_data)

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                clf_X_train = X_train
                clf_X_test = X_test

                print " - {}".format(name)
                clf.fit(clf_X_train, y_train)
                predictions_df = clf.predict(clf_X_test)
                ll = log_loss(
                    y_test, 'OutcomeType', predictions_df, possible_outcomes)

                print "   - score: %.5f" % ll
