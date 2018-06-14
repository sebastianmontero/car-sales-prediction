'''
Created on Jun 14, 2018

@author: nishilab
'''
import unittest
from utils import Utils


class Test(unittest.TestCase):


    def test_add_months_to_month_id(self):
        self.assertEqual(Utils.add_months_to_month_id(201101, 0), 201101, 'For adding 0 months')
        self.assertEqual(Utils.add_months_to_month_id(201101, 1), 201102, 'For adding 1 month')
        self.assertEqual(Utils.add_months_to_month_id(201111, 1), 201112, 'For adding 1 month close to end')
        self.assertEqual(Utils.add_months_to_month_id(201112, 1), 201201, 'For adding 1 month at the end of year')
        self.assertEqual(Utils.add_months_to_month_id(201101, 2), 201103, 'For adding 2 month')
        self.assertEqual(Utils.add_months_to_month_id(201111, 2), 201201, 'For adding 1 month')
        self.assertEqual(Utils.add_months_to_month_id(201101, 12), 201201, 'For adding 12 month')
        self.assertEqual(Utils.add_months_to_month_id(201101, 24), 201301, 'For adding 24 month')
        self.assertEqual(Utils.add_months_to_month_id(201101, 27), 201304, 'For adding 27 month')
        self.assertEqual(Utils.add_months_to_month_id(201103, 25), 201304, 'For adding 27 month')
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()