'''
Created on Apr 26, 2016

@author: Paul Reiners
'''
import unittest
import pandas as pd
from core.util import log_loss


class util_test(unittest.TestCase):
    def test_log_loss(self):
        truth_labels = ["0", "0", "0", "1", "1", "1"]
        predictions_col1 = [0.5, 0.1, 0.01, 0.9, 0.75, 0.001]
        predictions_col2 = [0.5, 0.9, 0.99, 0.1, 0.25, 0.999]
        predictions_df = pd.DataFrame({"0": predictions_col1, "1": predictions_col2})
        
        score = log_loss(truth_labels, "label", predictions_df, possible_labels=["0", "1"])
        error_tolerance = 0.0001
        self.assertAlmostEqual(1.881797068998267, score, delta=error_tolerance)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_log_loss']
    unittest.main()