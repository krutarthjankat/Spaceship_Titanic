import pandas as pd 
from sklearn.ensemble import  HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from warnings import simplefilter
simplefilter(action='ignore', category=Warning)

training_data_set = pd.read_csv("../data/processed_train.csv", header=0)
training_data_set.columns.values[0]='Sr_No'

X_train = training_data_set.drop('Transported',axis=1)
y_train = training_data_set['Transported']

lgb=LGBMClassifier(verbose=0)
hgb=HistGradientBoostingClassifier()
ann = MLPClassifier(hidden_layer_sizes=(16,8),max_iter=55)
cb = CatBoostClassifier(iterations=500, learning_rate=0.01,depth=10,verbose=0,allow_writing_files=False)

model=VotingClassifier(estimators=[('lgb',lgb),('hgb',hgb),('ann',ann),('cb',cb)],voting='soft',verbose=0)

model.fit(X_train,y_train)
joblib.dump(model,'../models/trained_model.joblib')

