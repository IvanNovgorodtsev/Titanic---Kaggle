import pandas as pd, sklearn.ensemble
from sklearn.tree import DecisionTreeRegressor

train = 'train.csv'
test = 'test.csv'
titanic_data = pd.read_csv(train)
titanic_test = pd.read_csv(test)

# model = sklearn.ensemble.GradientBoostingClassifier()
# model.fit(pd.get_dummies(titanic_data[['Pclass', 'Sex', 'Embarked']]), titanic_data['Survived'])
# prediction = model.predict(pd.get_dummies(titanic_test[['Pclass', 'Sex', 'Embarked']]))
# prediction[(titanic_test.Age <= 14) & (titanic_test.Pclass.isin([1, 2]))] = 1 
# pd.concat([titanic_test['PassengerId'], pd.DataFrame(prediction, columns=['Survived'])], axis=1).to_csv('submission.csv', index=False)

titanic = pd.concat([titanic_data,titanic_test], sort=False)
print(titanic.select_dtypes(include='float').head())