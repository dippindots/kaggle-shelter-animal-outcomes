'''
Created on May 25, 2016

@author: Paul Reiners

Based on
["Workflows in Python: Curating Features and Thinking Scientifically about
  Algorithms"]
(https://civisanalytics.com/blog/data-science/2015/12/23/workflows-in-python-curating-features-and-thinking-scientifically-about-algorithms/)
by Katie Malone
'''
from scripts.workflow.utils import \
    get_features_and_labels, get_names_of_columns_to_transform, hot_encoder


if __name__ == '__main__':
    features_df, labels_df = get_features_and_labels()
    print(features_df.columns.values)

    names_of_columns_to_transform = get_names_of_columns_to_transform()
    for feature in names_of_columns_to_transform:
        features_df = hot_encoder(features_df, feature)

    print(features_df.head())
