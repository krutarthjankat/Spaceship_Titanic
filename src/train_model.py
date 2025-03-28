sns.heatmap(training_data_set.isnull(), cbar=False, yticklabels=False)
meanimputer.fit(training_data_set.iloc[:, 5:6])
training_data_set.iloc[: , 5:6] = meanimputer.transform(training_data_set.iloc[:, 5:6])
meanimputer.fit(training_data_set.iloc[:, 7:12])
training_data_set.iloc[: , 7:12] = meanimputer.transform(training_data_set.iloc[:, 7:12])
    value_counts = non_null_values_train.value_counts(normalize=True)
    null_indices_train = training_data_set[col].isnull()
    # training_data_set.loc[null_indices_train, col] = np.random.choice(value_counts.index, size=null_indices_train.sum(), p=value_counts.values)
X = training_data_set.iloc[:, :-1]
y = training_data_set.iloc[:, -1]
y = lbl_enc.fit_transform(y)
X['CryoSleep'] = lbl_enc.fit_transform(X['CryoSleep']) 
X['VIP'] = lbl_enc.fit_transform(X['VIP']) 
X_test['CryoSleep'] = lbl_enc.fit_transform(X_test['CryoSleep']) 
X_test['VIP'] = lbl_enc.fit_transform(X_test['VIP']) 
samename_train = X['Firstname'].value_counts()
X['Same_Name'] = X['Firstname'].apply(lambda x: samename_train[x])
family_count_train = X['Surname'].value_counts()
X['Family_Size'] = X['Surname'].apply(lambda x: family_count_train[x])
group_count_train = X['Group'].value_counts()
X['Group_Size'] = X['Group'].apply(lambda x:group_count_train[x])
sc = StandardScaler().fit(X[scalable_features])
from catboost import CatBoostClassifier,Pool
model = CatBoostClassifier(iterations=500, learning_rate=0.01,depth=10,verbose=0) 
model.fit(X_train, y_train)
feature_importances = model.feature_importances_
pred = model.predict(X_validation)
model=LGBMClassifier(verbose=0)
model.fit(X_train,y_train)
feature_importances = model.feature_importances_
y_pred=model.predict(X_validation)
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train,y_train)
feature_importances = model.feature_importances_
y_pred=model.predict(X_validation)
model = HistGradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_validation)
from sklearn.model_selection import GridSearchCV
model = MLPClassifier(hidden_layer_sizes=(16,8),max_iter=52)
model = GridSearchCV(MLPClassifier(), param_grid, cv=3, scoring='accuracy')
model.fit(X_train,y_train)
y_pred=model.predict(X_validation)
print(model.best_params_)
X_train1 = tf.convert_to_tensor(X_train, dtype=tf.float64)
y_train1 = tf.convert_to_tensor(y_train, dtype=tf.float64)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 16, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 8, activation = 'relu'))
model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train1, y_train1,validation_data=[X_validation,y_validation],batch_size=32, epochs=60)
y_pred = model.predict(X_validation1)
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
labels = ['LightGBM','HistGB','Artificial Neural Network','XGBoost','Decision_Tree']
cb = CatBoostClassifier()
xgb = XGBClassifier()
# for clf, label in zip([lgb,hgb,ann_model,xgb,dt], labels):
#     scores = model_selection.cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
model=VotingClassifier(estimators=[('lgb',lgb),('hgb',hgb),('ann',ann),('cb',cb)],voting='hard',verbose=0)
model.fit(X_train,y_train)
pred=model.predict(X_validation)
y_final = model.predict(X_test)