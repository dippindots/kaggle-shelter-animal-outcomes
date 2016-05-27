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
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline
from sklearn.preprocessing import MinMaxScaler
import sklearn.tree

from core.preprocessing.feature_extraction_scaling import preprocess_age
from scripts.workflow.utils import \
    get_features_and_labels, get_names_of_columns_to_transform, hot_encoder


if __name__ == '__main__':
    features_df, labels_df = get_features_and_labels()
    print(labels_df.head(20))

    def label_map(y):
        if y == "Adoption":
            return 4
        elif y == "Died":
            return 3
        elif y == "Euthanasia":
            return 2
        elif y == "Return_to_owner":
            return 1
        else:
            # Transfer
            return 0
    labels_df = labels_df.applymap(label_map)
    print(labels_df.head())

    def transform_feature(df, column_name):
        unique_values = set(df[column_name].tolist())
        transformer_dict = {}
        for ii, value in enumerate(unique_values):
            transformer_dict[value] = ii

        def label_map(y):
            return transformer_dict[y]
        df[column_name] = df[column_name].apply(label_map)
        return df

    # list of column names indicating which columns to transform;
    # this is just a start!  Use some of the print( labels_df.head() )
    # output upstream to help you decide which columns get the
    # transformation
    names_of_columns_to_transform = get_names_of_columns_to_transform()
    for column in names_of_columns_to_transform:
        features_df = transform_feature(features_df, column)

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

    print(features_df.head())

    # remove the "date_recorded" column--we're not going to make use
    # of time-series data today
    features_df.drop("DateTime", axis=1, inplace=True)

    features_df.drop("OutcomeSubtype", axis=1, inplace=True)
    features_df.drop("Name", axis=1, inplace=True)

    print(features_df.columns.values)

    X = features_df.as_matrix()
    y = labels_df["OutcomeType"].tolist()

    clf = sklearn.linear_model.LogisticRegression()
    score = sklearn.cross_validation.cross_val_score(clf, X, y)
    print(score)

    clf = sklearn.tree.DecisionTreeClassifier()
    score = sklearn.cross_validation.cross_val_score(clf, X, y)
    print(score)

    clf = sklearn.ensemble.RandomForestClassifier()
    score = sklearn.cross_validation.cross_val_score(clf, X, y)
    print(score)

    for feature in names_of_columns_to_transform:
        features_df = hot_encoder(features_df, feature)
        features_df.drop(feature, axis=1, inplace=True)

    print(features_df.head())

    X = features_df
    select = sklearn.feature_selection.SelectKBest(k=100)
    selected_X = select.fit_transform(X, y)

    print(selected_X.shape)

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

    # fit your pipeline on X_train and y_train
    pipeline.fit(X_train, y_train)
    # call pipeline.predict() on your X_test data to make a set of test
    # predictions
    y_prediction = pipeline.predict(X_test)
    # test your predictions using sklearn.classification_report()
    report = sklearn.metrics.classification_report(y_test, y_prediction)
    # and print the report
    print(report)

    parameters = dict(feature_selection__k=[100, 200],
                      random_forest__n_estimators=[50, 100, 200],
                      random_forest__min_samples_split=[2, 3, 4, 5, 10])

    cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

    cv.fit(X_train, y_train)
    y_predictions = cv.predict(X_test)
    report = sklearn.metrics.classification_report(y_test, y_predictions)
    print(report)
