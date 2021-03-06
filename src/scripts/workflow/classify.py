'''
Created on May 25, 2016

@author: Paul Reiners

Based on
* ["Workflows in Python: Getting data ready to build models"]
  (https://civisanalytics.com/blog/data-science/2015/12/17/workflows-in-python-getting-data-ready-to-build-models/)
* ["Workflows in Python: Curating Features and Thinking Scientifically about
    Algorithms"]
  (https://civisanalytics.com/blog/data-science/2015/12/23/workflows-in-python-curating-features-and-thinking-scientifically-about-algorithms/)
* ["Workflows in Python: Using Pipeline and GridSearchCV for More Compact and
    Comprehensive Code"]
  (https://civisanalytics.com/blog/data-science/2016/01/06/workflows-python-using-pipeline-gridsearchcv-for-compact-code/)
by Katie Malone
'''
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.grid_search
import sklearn.pipeline
from sklearn.preprocessing import MinMaxScaler
from core.learning.performance_metrics import bundle_predictions
from core.learning.performance_metrics import log_loss

from core.preprocessing.feature_extraction_scaling \
    import preprocess_age, get_is_named, extract_date_time_features, \
    is_dangerous
import pandas as pd
from scripts.workflow.utils import \
    get_features_and_labels, get_names_of_columns_to_transform


if __name__ == '__main__':
    features_df, labels_df = get_features_and_labels()

    # list of column names indicating which columns to transform;
    # this is just a start!  Use some of the print( labels_df.head() )
    # output upstream to help you decide which columns get the
    # transformation
    names_of_columns_to_transform = get_names_of_columns_to_transform()

    def transform_age_upon_outcome(age_upon_outcome):
        transformed_age_upon_outcome = age_upon_outcome.apply(preprocess_age)
        transformed_age_upon_outcome = transformed_age_upon_outcome.fillna(
            transformed_age_upon_outcome.mean())

        mms = MinMaxScaler()
        transformed_age_upon_outcome = mms.fit_transform(
            transformed_age_upon_outcome.reshape(-1, 1))

        return transformed_age_upon_outcome
    features_df['AgeuponOutcome'] = transform_age_upon_outcome(
        features_df['AgeuponOutcome'])

    # remove the "DateTime" column--we're not going to make use
    # of time-series data today
    features_df = extract_date_time_features(features_df)
    features_df.drop("DateTime", axis=1, inplace=True)

    features_df.drop("OutcomeSubtype", axis=1, inplace=True)
    features_df['Name'] = features_df['Name'].fillna('')
    features_df['IsNamed'] = features_df['Name'].apply(get_is_named)
    features_df.drop("Name", axis=1, inplace=True)
    features_df['IsDangerous'] = features_df['Breed'].apply(is_dangerous)

    X = features_df.as_matrix()
    y = labels_df["OutcomeType"].tolist()

    for categorical_column in names_of_columns_to_transform:
        features_df[categorical_column] = features_df[
            categorical_column].astype('category')
    features_df = pd.get_dummies(
        features_df, columns=names_of_columns_to_transform)

    X = features_df

    #########################################################################
    # Using Pipeline and GridSearchCV for More Compact and Comprehensive Code
    #########################################################################
    select = sklearn.feature_selection.SelectKBest(k=100)
    clf = sklearn.ensemble.RandomForestClassifier()

    steps = [('feature_selection', select),
             ('random_forest', clf)]

    pipeline = sklearn.pipeline.Pipeline(steps)

    X_train, X_test, y_train, y_test = \
        sklearn.cross_validation.train_test_split(
            X, y, test_size=0.33, random_state=42)

    parameters = dict(feature_selection__k=[50, 100, 200],
                      random_forest__n_estimators=[50, 100, 200, 400],
                      random_forest__min_samples_split=[2, 3, 4, 5, 10, 20])

    cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

    cv.fit(X_train, y_train)
    print cv.best_params_
    # http://stackoverflow.com/q/37501450/7648

    y_predictions = cv.predict_proba(X_test)

    y_prediction_df = bundle_predictions(y_predictions)
    possible_outcomes = [
        'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    ll = log_loss(
        y_test, 'OutcomeType', y_prediction_df, possible_outcomes)

    # BEST: 0.80022
    print "score: %.5f" % ll
