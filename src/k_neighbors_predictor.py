'''
Created on Apr 27, 2016

MyLLScore:     13.99550     
KaggleLLScore: 

@author: Paul Reiners
'''
from predictor_base import PredictorBase
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 

class KNeighborsPredictor(PredictorBase):
    '''
    Uses k-nearest neighbors.
    '''


    def predict(self, X_train, X_test, y_train):
        neigh = KNeighborsClassifier()
        neigh.fit(X_train, y_train)
        predictions = neigh.predict(X_test)
        test_n = len(X_test)
        zeroes = [0.0] * test_n
        predictions_data = {
                            'ID': range(test_n), 'Adoption': zeroes, 'Died': zeroes, 
                            'Euthanasia': zeroes, 'Return_to_owner': zeroes, 'Transfer': zeroes}
        predictions_df = pd.DataFrame(predictions_data)
        for i in range(test_n):
            prediction = predictions[i]
            predictions_df.loc[i, (prediction)] = 1.0
        
        return predictions_df
    
        
