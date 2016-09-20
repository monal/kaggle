
# Rate My Professor - Team Furonics

The model used was **Ridge Regression**. Features included - comments, tags, grades, interest and textbookuse. 

Takeaways - FeatureUnion and Pipelines, clipping.


```python
import pandas as pd
import numpy as np
import re
```

### Read in the train and test data


```python
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
```

### Feature 1: Comments - Missing comments are replaced with empty strings


```python
train_data['comments'] = train_data['comments'].fillna('')
test_data['comments'] = test_data['comments'].fillna('')
```

### Feature 2: Grades - Encode categorical values of grades


```python
def grade_encoder(dataset):
    grade_encoding = pd.get_dummies(dataset['grade'])
    grade_encoding_as_array = np.asarray(grade_encoding)
    
    grade_encoding_as_df = [grade_encoding_as_array[x].tostring() for x in xrange(len(dataset))]
    dataset['grade_encoding'] = pd.DataFrame(grade_encoding_as_df)

grade_encoder(train_data)
grade_encoder(test_data)
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
```

### Split training data - 75% for training, 25% for validation


```python
train, test = train_test_split(trainData)
```

### Using sklearn's FeatureUnion to parallely encode categorical data in Features
#### FeatureUnion useful when
#### 1) Dataset has heterogenous data types
#### 2) Different columns require different pre-processing Pipelines

### FeatureUnion combines several 'transformer' objects 

### Transformer class for comments


```python
class textExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, factor):
        self.factor = factor
        
    def transform(self,data):
        return np.asarray(data[self.factor])

    def fit(self, *_):
        return self
```

### Transformer class for tags
### Each tag is first cleaned using regex and then encoded based on a separator. 


```python
parserForTag = lambda x : re.sub("[^\Sa-zA-Z]", "", x).replace("\"","").replace("[","").replace("]","").replace("\'","").replace('.',"").replace('?',"").lower().split(",")

class tagExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, factor):
        self.factor = factor
    
    def transform(self, data):
        return np.asarray(data[self.factor].map(parserForTag).str.join(sep='*').str.get_dummies(sep='*'))

    def fit(self, *_):
        return self
```


```python
class interestExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, factor):
        self.factor = factor
        
    def transform(self,data):
        return pd.get_dummies(data[self.factor])
        
    def fit(self, *_):
        return self
```


```python
class gradeExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, factor):
        self.factor = factor
        
    def transform(self,data):
        outputGrades = []
        for content in data[self.factor]:
            outputGrades.append(np.fromstring(content))
        return outputGrades
    
    def fit(self, *_):
        return self
```

### Comments are converted to a matrix of TF-IDF features. 30000 features are used in the final model

### All the individual transformer objects are concatenated to a single transformer using the FeatureUnion estimator 


```python
comments_featurizer = Pipeline([
        ('comments_extractor', textExtractor('comments')),
        ('comments_tdidf', TfidfVectorizer(max_df=0.95, 
                                           min_df=1,  
                                           max_features=30000,
                                           ngram_range = (1,3)))
    ])

tags_featurizer = Pipeline([
        ('tags_extractor', tagExtractor('tags'))
    ])

interest_featurizer = Pipeline([
        ('interest_extractor', interestExtractor('interest'))
    ])

grade_featurizer = Pipeline([
        ('grade_extractor', gradeExtractor('gradeContent'))
    ])

textBook_featurizer = Pipeline([
        ('textBook_extractor', interestExtractor('textbookuse'))
    ])

features = FeatureUnion([        
        ('comments_features', comments_featurizer),
        ('interest_features', interest_featurizer),        
        ('tag_features', tags_featurizer),
        ('grade_features', grade_featurizer)
    ])

pipeline = Pipeline([
  ('feature_union', features),
  ('regression', Ridge())
])
```

    Fitting 3 folds for each of 6 candidates, totalling 18 fits


### Tuning the hyperparameters and cross-validation using GridSearchCV


```python
alphas = np.array([3,1,0.1]) 
max_iter = np.array([800, 1300])
```


```python
cv = GridSearchCV(
    pipeline, param_grid=dict(regression__alpha=alphas, regression__max_iter=max_iter), n_jobs=-1, verbose=10
).fit(train[['comments', 'interest','tags','grade','gradeContent','textbookuse']], train['quality'])
```


```python
# Output the best score, this is based on held out data in cross validation
print("R Squared: {}".format(cv.best_score_))

# Output the Mean Squared Error using our held out training data
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test['quality'], cv.predict(test[['comments','interest','tags','grade','gradeContent','textbookuse']]))
print("MSE: {}".format(mse))
```


```python
# Make training predictions
predictions = cv.predict(testData[['comments', 'interest','tags','grade','gradeContent','textbookuse']])

# Lets take a quick look at the predictions to make sure they are sensible, seems like it
predictions
```

### Any score less than 2 and above 10 are clipped to 2 and 10 respectively. This small hack reduced the MSE by ~0.06. Done using the clip function in numpy!


```python
# Finally lets write out the predictions with their id's

with open('predictions.csv', 'w') as f:
    f.write("id,quality\n")
    for row_id, prediction in zip(testData['id'], np.clip(predictions, 2, 10)):
        f.write('{},{}\n'.format(row_id, prediction))
```
