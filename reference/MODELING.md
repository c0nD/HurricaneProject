# Data Modeling Guide

This document provides step-by-step instructions on how to implement our model selection process using the provided Python code. The code is designed to tune the parameters for our regression model.

# Step 1: Import Required Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

# Step 2: Load the Dataset
```python
df = pd.read_csv('data/pacific_cleaned.csv')
```

# Step 3: Import Model and Model Selection Tools, and prepare target variables

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

df['Next_Latitude'] = df['Latitude'].shift(-1)
df['Next_Longitude'] = df['Longitude'].shift(-1)

# Drop any NaN values
df = df.dropna()

X = df[['Latitude', 'Longitude']]
y = df[['Next_Latitude', 'Next_Longitude']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

# Step 4: Create Parameter Grid for Tuning, Then Use GridSearchCV to Find Best Parameter Values

```python
# this is the regression model we have chosen!
model = RandomForestRegressor()

# these are the parameter that we are tuning
param_grid = {
    'n_estimators': [100, 200, 300], # decides the number of trees in our forest
    'max_depth': [None, 10, 20], # the maximum depth of a tree
    'min_samples_split': [2, 5, 10], # the minimum amount of samples needed to split an internal node in the tree
    'min_samples_leaf': [1, 2, 4] # the minimum amount of samples required to be at a leaf node
}

# this is what we'll use to tune our parameters
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train)

# this tell us the best model found by grid-searching
best_model = grid_search.best_estimator_
```

# Step 4: Obtaining the Best Parameters and Visualizing the Mean Scores of the Training and Test Data
```python
# this tells us the best values for each of the parameters we tuned for our model
best_params = grid_search.best_params_
grid_scores = grid_search.cv_results_
print("Best parameters: ", best_params)

# filtering scores in order to visualize our scores correctly
filtered_scores = []
for i in range(len(grid_scores['params'])):
    if all(grid_scores['params'][i][k] == best_params[k] 
           for k in best_params if k != 'n_estimators'):
        filtered_scores.append((grid_scores['params'][i]['n_estimators'], grid_scores['mean_test_score'][i]))

filtered_scores_2 = []
for i in range(len(grid_scores['params'])):
    if all(grid_scores['params'][i][k] == best_params[k] 
           for k in best_params if k != 'n_estimators'):
        filtered_scores_2.append((grid_scores['params'][i]['n_estimators'], grid_scores['mean_train_score'][i]))

filtered_scores.sort(key=lambda x: x[0])
filtered_scores_2.sort(key=lambda x: x[0])

n_estimators, mean_test_scores = zip(*filtered_scores)
n_estimators, mean_train_scores = zip(*filtered_scores_2)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators, mean_train_scores, label='Mean Train Score', color='red')
plt.plot(n_estimators, mean_test_scores, label='Mean Test Score', color='blue')

plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Mean Test Score')
plt.title('Grid Search: Mean Test Score vs. n_estimators')
plt.legend(loc='best')
plt.grid()
plt.show()
```



