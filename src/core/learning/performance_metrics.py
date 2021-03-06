'''
Created on May 2, 2016

Performance metric used in Kaggle competition.

@author: Paul Reiners
'''
from math import log
import pandas as pd


def log_loss(truths, label_col_name, predictions_df, possible_labels):
    ''' Function to measure log loss of a prediction.

    Parameters
    ==========
    truths          : numpy.ndarray
                      the ground truth
    label_col_name  : str
                      the name of the column you're trying to predict
    predictions_df  : pandas.core.frame.DataFrame
                      your predictions
    possible_labels : list
                      possible labels'''
    n = len(truths)
    total = 0.0
    for i in range(n):
        truth = truths[i]
        for possible_label in possible_labels:
            if truth == possible_label:
                y = 1
            else:
                y = 0
            prediction = predictions_df.iloc[i]
            p = prediction[truth]
            p = max(min(p, 1 - 1e-15), 1e-15)
            total += y * log(p)

    return -1.0 / n * total


def bundle_predictions(predictions):
    n = len(predictions)
    predictions = predictions.transpose()
    predictions_data = {
        'ID': range(1, n + 1), 'Adoption': predictions[0],
        'Died': predictions[1], 'Euthanasia': predictions[2],
        'Return_to_owner': predictions[3], 'Transfer': predictions[4]}
    predictions_df = pd.DataFrame(predictions_data)

    return predictions_df
