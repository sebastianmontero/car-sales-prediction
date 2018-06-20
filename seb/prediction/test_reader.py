'''
Created on Jun 19, 2018

@author: nishilab
'''
import unittest
from reader import Reader


class Test(unittest.TestCase):


    def test_add_scaler_fit_domain_values(self):
        features = ['interest_rate']
        reader = Reader(13, 20, features)
        reader.scaler_fit_domain_values = {
            'interest_rate': [1.0,10.0],
            'exchange_rate': [8.0, 18.0]
        }
        
        data=[[4.5],[3.7]]
        data = reader._add_scaler_fit_domain_values(features, data)
        self.assertSequenceEqual(data.tolist(),[[4.5],[3.7],[1.0],[10.0]])
        
        features = ['interest_rate', 'exchange_rate']
        reader = Reader(13, 20, features)
        reader.scaler_fit_domain_values = {
            'interest_rate': [1.0,10.0],
            'exchange_rate': [8.0, 18.0]
        }
        
        data=[[4.5,15.7],[3.7,14.8]]
        data = reader._add_scaler_fit_domain_values(features, data)
        self.assertSequenceEqual(data.tolist(),[[4.5,15.7],[3.7,14.8],[1.0, 8.0],[10.0,18.0]])
        
        features = ['interest_rate', 'exchange_rate']
        reader = Reader(13, 20, features)
        reader.scaler_fit_domain_values = {
            'interest_rate': [1.0,10.0],
            'exchange_rate': [8.0]
        }
        
        data=[[1,2],[1,2]]
        data = reader._add_scaler_fit_domain_values(features, data)
        self.assertSequenceEqual(data.tolist(),[[1,2],[1,2],[1.0, 8.0],[10.0,2.0]])
        
        features = ['interest_rate', 'exchange_rate']
        reader = Reader(13, 20, features)
        reader.scaler_fit_domain_values = {
            'interest_rate': [1.0,10.0]
        }
        
        data=[[1,2],[1,2]]
        data = reader._add_scaler_fit_domain_values(features, data)
        self.assertSequenceEqual(data.tolist(),[[1,2],[1,2],[1.0, 2.0],[10.0, 2.0]])
        
        features = ['interest_rate', 'exchange_rate']
        reader = Reader(13, 20, features)
        reader.scaler_fit_domain_values = {
            'exchange_rate': [8.0, 18.0, 20.0]
        }
        
        data=[[4.5,15.7],[3.7,14.8]]
        data = reader._add_scaler_fit_domain_values(features, data)
        self.assertEqual(len(data), 5)
        self.assertSequenceEqual(data.tolist()[:2],[[4.5,15.7],[3.7,14.8]])
        for i in range(2,5):
            self.assertIn(data[i][0], [4.5, 3.7])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()