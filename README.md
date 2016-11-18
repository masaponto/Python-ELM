# Python-ELM
Extreme Learning Machine implemented in Python3 with scikit-learn  

This module has 3-type ELM  
- elm.py is general 3-step model ELM   
- ecob_elm.py is equality constrained optimization based ELM  
  - http://www.ntu.edu.sg/home/egbhuang/pdf/ELM-Unified-Learning.pdf
- ml_elm is multi layer ELM  
  - http://www.ntu.edu.sg/home/egbhuang/pdf/ieee-is-elm.pdf

## Require
- scikit-learn: 0.18.1  
- numpy: 1.10.4 and up  

## How to install 
```
pip install git+https://github.com/masaponto/python-elm
```

## Usage

### Basic

```python
from elm import ELM
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

db_name = 'australian'

data_set = fetch_mldata(db_name)
data_set.data = preprocessing.normalize(data_set.data)

X_train, X_test, y_train, y_test = train_test_split(
    data_set.data, data_set.target, test_size=0.4)

elm = ELM(hid_num=10).fit(X_train, y_train)

print("ELM Accuracy %0.3f " % elm.score(X_test, y_test))
```

### For cross-validation

```python
from elm import ELM
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import ShuffleSplit, KFold, cross_val_score

db_names = ['australian', 'iris']
hid_nums = [10, 20, 30]

for db_name in db_names:
    print(db_name)
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.normalize(data_set.data)

    for hid_num in hid_nums:
        print(hid_num, end=' ')
        e = ELM(hid_num)

        ave = 0
        for i in range(10):
            cv = KFold(n_splits=5, shuffle=True)
            scores = cross_val_score(
                e, data_set.data, data_set.target, cv=cv, scoring='accuracy', n_jobs=-1)
            ave += scores.mean()

        ave /= 10

        print("Accuracy: %0.3f " % (ave))
```
