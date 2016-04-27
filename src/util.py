'''
Created on Apr 26, 2016

@author: Paul Reiners
'''
from math import log

def log_loss(truth_df, label_col_name, predictions_df, possible_labels):
    n = len(truth_df)
    total = 0.0
    for i in range(n):
        truth = truth_df[label_col_name].iloc[i]
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
