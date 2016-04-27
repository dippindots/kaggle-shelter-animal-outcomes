'''
Created on Apr 27, 2016

MyLLScore:     14.37823     
KaggleLLScore: 13.94696

@author: Paul Reiners
'''
from predictor_base import PredictorBase
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 

class KNeighborsPredictor(PredictorBase):
    '''
    Uses k-nearest neighbors.
    '''
    
    def fit(self, X_train, y_train):
        self.neigh = KNeighborsClassifier()
        self.neigh.fit(X_train, y_train)
    

    def predict(self, X_test):
        predictions = self.neigh.predict_proba(X_test)
        predictions = predictions.transpose()
        n = len(X_test)
        predictions_data = {
                            'ID': range(1, n + 1), 'Adoption': predictions[0], 'Died': predictions[1], 
                            'Euthanasia': predictions[2], 'Return_to_owner': predictions[3], 'Transfer': predictions[4]}
        predictions_df = pd.DataFrame(predictions_data)
        
        return predictions_df
    
        
