import sklearn
sklearn.__version__


import numpy as np
import pandas as pd
np.random.seed(42) # To stabilise the output during runs


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


train_data.head()


train_data.info()


train_data.describe()


train_data['Survived'].value_counts()


train_data['Pclass'].value_counts()


train_data['Sex'].value_counts()


train_data['Embarked'].value_counts()


train_data_copy = train_data.copy()
# Use the "copy()" function call to avoid changing the original data (which in this case is "train_data")
# We have to poke around with transformations using the copy of train_data and see what we can archieve
train_data_copy.head()


from sklearn.base import BaseEstimator, TransformerMixin

# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='median')
# Setting strategy to median indicates that we want to convert the NaN values to the median of the numerical attribute


impute.fit(train_data_copy[["Age"]])
train_data_copy[["Age"]] = impute.transform(train_data_copy[["Age"]])
train_data_copy.info()


cat_impute = MostFrequentImputer()
cat_impute.fit(train_data_copy[["Embarked"]])
train_data_copy[["Embarked"]] = cat_impute.transform(train_data_copy[["Embarked"]])
train_data_copy.info()


train_data_copy.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)


from sklearn.preprocessing import StandardScaler

num_attribs = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
# Pclass is a category but it's already in numerical so scaling it would be a good idea than labeling it in this case
scaler = StandardScaler()
train_data_copy[num_attribs] = scaler.fit_transform(train_data_copy[num_attribs])
train_data_copy.head()


from sklearn.preprocessing import LabelEncoder

labeler = LabelEncoder()
for cats in ["Sex", "Embarked"]: # cats is just short for categories, not actual cats ;)
    train_data_copy[cats] = labeler.fit_transform(train_data_copy[cats])


train_data_copy.head()


from sklearn.pipeline import Pipeline # how ironic, importing Pipeline from pipeline ;)
from sklearn.preprocessing import OneHotEncoder
# In this one we will use OneHotEncoder instead of LabelEncoder as OneHotEncoder tends to do a better job than LabelEncoder

# We will make 2 Pipelines, one for Numerical Attributes and the other for Categorial Attributes

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    # The name "imputer" can be set to any string e.g "impute" or "whatever"
    ('scaler', StandardScaler())
])

# I had already imported these packages before so there's no need for repetition

cat_pipeline = Pipeline([
    ('imputer', MostFrequentImputer()), # from the class we had made earlier
    ('encoding', OneHotEncoder(sparse=False))
    # Setting sparse to False prevents the OneHotEncoder from returning a Scipy sparse matrix
])



y_train = train_data['Survived']
# We will drop the "Survived" Attribute in our train set before transforming it with our Pipeline since it may cause some
# problems with our test set when transforming the values in test set because the test set has no "Survived" Attribute
train_data.drop('Survived', axis=1, inplace=True)


from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
    ('num_pipeline', num_pipeline, ["Age", "SibSp", "Parch", "Fare"]),
    # We just choose the numerical attributes we want
    ('cat_pipeline', cat_pipeline, ["Pclass", "Sex", "Embarked"])
    # Here, we just choose the categorial attributes we want and here Pclass works better as a category than a number
])

X_train = full_pipeline.fit_transform(train_data)

# The transformation returns our data as a numpy array
# Only the attributes of numbers and categories that we have specified in the full_pipeline (which are just 7) will
# be present in the data so there is really no need of dropping attributes like we did before in our "Regular
# Transformation" detour because they have been automatically dropped.

X_train[:5] # Our numpy array's head, Similar with Pandas .head() function call :)
# Voila...


# As you saw, Pipelines are easily managable than the whole transformation process we had earlier and Pipelines also
# take less time to set up

# Let's see how our data looks like as a DataFrame

# Pclass: 1, 2, 3
# Sex: Female, Male
# Embarked: S,C, Q

columns = ["Age", "SibSp", "Parch", "Fare", 'Pclass-1', 'Pclass-2', 'Pclass-3', 
           'Sex-Female', 'Sex-Male', 'Embarked-C', 'Embarked-Q', 'Embarked-S']

pd.DataFrame(X_train, columns=columns).head()


# We will start by using the Stochastic Gradient Descent Classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)


X_test = full_pipeline.transform(test_data)
predictions = sgd_clf.predict(X_test)


#We convert our predictions to a csv file
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('predictions.csv')

# Then you can submit the csv file


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier().fit(X_train, y_train)
dt_pred = dt_clf.predict(X_train)
accuracy_score(y_train, dt_pred)


from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train, scoring='accuracy', cv=10).mean()


# And if you would like to get some predictions so that you can compare them with the y_train set you can simply:

from sklearn.model_selection import cross_val_predict

predictions = cross_val_predict(sgd_clf, X_train, y_train, cv=10)
accuracy_score(y_train, predictions)


# If you would like to get some Decision Functions you can simply:

predictions = cross_val_predict(sgd_clf, X_train, y_train, cv=10, method='decision_function')
predictions[:5]


# Let's try another model
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
cross_val_score(knn_clf, X_train, y_train, scoring='accuracy', cv=10).mean()


# The other model
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
cross_val_score(rnd_clf, X_train, y_train, scoring='accuracy', cv=10).mean()


# We can tweak the n_neighbors parameter in KNeighborsClassifier this way:

for number in range(1, 12):
    knn_looped = KNeighborsClassifier(n_neighbors=number).fit(X_train, y_train)
    score = cross_val_score(knn_looped, X_train, y_train, scoring='accuracy', cv=10).mean()
    print(number, score)


from sklearn.model_selection import GridSearchCV

params = [
    {'n_estimators': [100, 200, 300], 'max_features': [8, 9, 10, 11]}
] # These are the parameters that we put in Random Forest model and test each and every combination

rnd_clf = RandomForestClassifier(random_state=42)
grid_rnd_clf = GridSearchCV(rnd_clf, params, cv=3, return_train_score=True, scoring='accuracy')
# The grid search takes the algorithm, parameters, folds (number of trainings)
grid_rnd_clf.fit(X_train, y_train)
# Fitting our Grid Search Model make take some time, maybe a few seconds


grid_rnd_clf.best_score_


# Let's see the best combinations/parameters
grid_rnd_clf.best_params_


# Let's see the whole parameters and their scores
data = grid_rnd_clf.cv_results_
for a, b in zip(data['mean_test_score'], data['params']):
    print(a, b)


rnd_clf = grid_rnd_clf.best_estimator_ # grid_rnd_clf.best_estimator_ is our model
cross_val_score(rnd_clf, X_train, y_train, cv=10, scoring='accuracy').mean()


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'n_estimators': randint(low=100, high=200), 'max_features': randint(low=6, high=10)}


rnd_clf2 = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(rnd_clf2, param_distributions = param_distribs, n_iter=10, cv=3, scoring='accuracy',
                               random_state=42)
rnd_search.fit(X_train, y_train)

# Too much randomness on this one, Ehh :)


rnd_search.best_score_, rnd_search.best_params_


# Let's see the whole parameters and their scores
data = rnd_search.cv_results_
for a, b in zip(data['mean_test_score'], data['params']):
    print(a, b)


rnd_clf = rnd_search.best_estimator_ # rnd_search.best_estimator_ is our model
cross_val_score(rnd_clf, X_train, y_train, cv=10, scoring='accuracy').mean()



