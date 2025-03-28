import joblib
import pandas as pd 
from sklearn.metrics import classification_report

model = joblib.load('../models/trained_model.joblib')
testing_data_set = pd.read_csv("../data/processed_test.csv", header=0)
testing_data_set.columns.values[0]='Sr_No'

X_test = testing_data_set.drop('Transported',axis=1)
y_test = testing_data_set['Transported']

pred=model.predict(X_test)
print(classification_report(y_test, pred))