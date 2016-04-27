'''
Created on Apr 27, 2016

MyLLScore:     20.61577  
KaggleLLScore: 20.25113

@author: Paul Reiners
'''
from predictor_base import PredictorBase
import pandas as pd 

class BaseLinePredictor(PredictorBase):
    '''
    All adopted benchmark
    '''


    def predict(self, X_train, X_test, y_train):
        test_n = len(X_test)
        zeroes = [0] * test_n
        predictions_data = {
                            'ID': test_n, 'Adoption': [1] * test_n, 'Died': zeroes, 
                            'Euthanasia': zeroes, 'Return_to_owner': zeroes, 'Transfer': zeroes}
        predictions_df = pd.DataFrame(predictions_data)
        
        return predictions_df
