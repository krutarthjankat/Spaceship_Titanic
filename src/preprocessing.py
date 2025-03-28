import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_set = pd.read_csv("../data/data.csv", header=0)

## imputing missing numeric data in training set
meanimputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
meanimputer.fit(data_set.iloc[:, 5:6])
data_set.iloc[: , 5:6] = meanimputer.transform(data_set.iloc[:, 5:6])
meanimputer.fit(data_set.iloc[:, 7:12])
data_set.iloc[: , 7:12] = meanimputer.transform(data_set.iloc[:, 7:12])
data_set = data_set.dropna()


## Encoding binary data
lbl_enc = LabelEncoder()
data_set['Transported'] = lbl_enc.fit_transform(data_set['Transported'])
data_set['CryoSleep'] = lbl_enc.fit_transform(data_set['CryoSleep']) 
data_set['VIP'] = lbl_enc.fit_transform(data_set['VIP']) 

# Feature Engineering
data_set['Deck'] = data_set['Cabin'].apply(lambda x: x.split('/')[0] if isinstance(x, str) and '/' in x else x)
data_set['Side'] = data_set['Cabin'].apply(lambda x: (x.split('/')[2]=='P') if isinstance(x, str) and '/' in x else x)
data_set['Firstname'] = data_set['Name'].apply(lambda x: x.split(' ')[0] if isinstance(x, str) and ' ' in x else 1)
samename_train = data_set['Firstname'].value_counts()
data_set['Same_Name'] = data_set['Firstname'].apply(lambda x: samename_train[x])
data_set['Surname'] = data_set['Name'].apply(lambda x: x.split(' ')[1] if isinstance(x, str) and ' ' in x else 1)
family_count_train = data_set['Surname'].value_counts()
data_set['Family_Size'] = data_set['Surname'].apply(lambda x: family_count_train[x])
data_set['Group'] = pd.to_numeric(data_set['PassengerId'].apply(lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else x))
group_count_train = data_set['Group'].value_counts()
data_set['Group_Size'] = data_set['Group'].apply(lambda x:group_count_train[x])
data_set['Total_Spending'] = data_set['RoomService']+data_set['FoodCourt']+data_set['ShoppingMall']+data_set['Spa']+data_set['VRDeck']
data_set['VR_Spa'] = data_set['Spa']+data_set['VRDeck']
data_set['Room_Food'] = data_set['RoomService']+data_set['FoodCourt']
data_set['VR_Spa_Room_Food'] = data_set['RoomService']+data_set['FoodCourt']+data_set['Spa']+data_set['VRDeck']
data_set = data_set.drop(['Cabin','PassengerId','Name','Firstname','Surname','Group'],axis=1)

# Feature Scaling
scalable_features = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Group_Size','Total_Spending','Family_Size','VR_Spa','Room_Food','VR_Spa_Room_Food','Same_Name']
sc = StandardScaler().fit(data_set[scalable_features])
data_set[scalable_features]=sc.transform(data_set[scalable_features])

# One Hot Encoding the Categorical data 
data_set = pd.get_dummies(data_set, columns=['HomePlanet','Destination','Deck'])*1.0
data_set.columns = data_set.columns.str.replace(' ', '_') 
data_set.columns = data_set.columns.str.replace('.', '_') 
print(data_set.head())


train, test = train_test_split(data_set, test_size = 0.2, random_state=42)

train.to_csv("../data/processed_train.csv")
test.to_csv("../data/processed_test.csv")