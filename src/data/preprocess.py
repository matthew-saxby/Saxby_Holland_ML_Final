import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

#load in data
from load_data import diabetes

#change male and female to 0 and 1
gender_map = {'Male': 0, 
              'Female': 1,
              'Other':2
              }

diabetes['gender'] = diabetes['gender'].replace(gender_map) 

#change smoking history
smoking_map = {'No Info':0,
               'never':1,
               'ever':2,
               'former':3,
               'not current':3,
               'current':4
               }

diabetes['smoking_history'] = diabetes['smoking_history'].replace(smoking_map) 

diabetes.to_csv('diabetes_processed.csv', index=False)

print(diabetes.head())
