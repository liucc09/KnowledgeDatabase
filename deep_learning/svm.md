- [linear svm](#linear-svm)
    - [Grid Search](#grid-search)
    - [Training](#training)
    - [Display results](#display-results)

# linear svm

```python
import time
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV

 
df_train = pd.read_csv('/home/liucc/data/new_data/train_set.csv')
#df_test = pd.read_csv('./test_set.csv')
 
df_train.drop(columns = ['article','id'], inplace = True)
#df_test.drop(columns = ['article'], inplace = True)

texts = df_train['word_seg'].values
labels = (df_train['class']-1).values

num = len(texts)
num_val = math.ceil(num*0.1)

x_val = texts[:num_val]
y_val = labels[:num_val]

x_train = texts[num_val:]
y_train = labels[num_val:]

vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
xv_train = vectorizer.fit_transform(x_train)

```

## Grid Search


```python
lin_clf = svm.SVC(kernel='linear')
grid = GridSearchCV(lin_clf,param_grid={"C":[0.1,1,10]})
grid.fit(xv_train[:1000,:],y_train[:1000])
print(grid.best_params_,grid.best_score_)
```

>{'C': 10} 0.631


## Training


```python
time_start = time.time()

lin_clf = svm.SVC(kernel='linear',probability=True)
lin_clf.fit(xv_train,y_train)
 
time_end = time.time()
print((time_end - time_start)/60)
```
## Display results

```python
from sklearn.metrics import classification_report
y_pred = lin_clf.predict(xv_train)
print(classification_report(y_train, y_pred)) 
```

id | precision  |  recall | f1-score  | support
:---:|:---:|:---:|:---:|:---:
0   | 0.98  | 0.95  |0.96   |   4827
1   | 0.99  |   0.99   |   0.99|  2617
2| 0.99|1.00|0.99|7488
3| 0.99|0.99|0.99|3447
4| 0.99|0.98|0.98|2140
5| 0.99|0.99|0.99|6198
6| 0.98|0.97|0.98|2717
7| 0.98|0.98|0.98|6289
8| 0.99|0.99|0.99|6902
9| 0.98|0.98|0.98|4474
10| 0.98|0.98|0.98|3225
11| 0.98|0.98|0.98|4778
12| 0.98|0.99|0.98|7113
13| 0.98|0.98|0.98|6081
14| 0.99|1.00|0.99|6747
15| 0.98|0.97|0.97|2905
16| 0.97|0.98|0.98|2793
17| 0.99|0.99|0.99|6368
18| 0.98|0.97|0.97|4940
avg / total| 0.98|0.98|0.98| 92049

```python
xv_val = vectorizer.transform(x_val)
y_pred = lin_clf.predict(xv_val)
print(classification_report(y_val, y_pred)) 
```

id| precision|recall | f1-score |  support
:---:|:---:|:---:|:---:|:---:
|  0| 0.69|0.56|0.62| 548
|  1| 0.79|0.80|0.79| 284
|  2| 0.89|0.92|0.91| 825
|  3| 0.87|0.84|0.85| 377
|  4| 0.79|0.79|0.79| 229
|  5| 0.94|0.89|0.91| 690
|  6| 0.75|0.69|0.72| 321
|  7| 0.71|0.78|0.75| 683
|  8| 0.92|0.93|0.93| 773
|  9| 0.74|0.72|0.73| 489
| 10| 0.66|0.66|0.66| 346
| 11| 0.72|0.64|0.68| 548
| 12| 0.70|0.81|0.75| 794
| 13| 0.75|0.86|0.80| 659
| 14| 0.91|0.91|0.91| 764
| 15| 0.81|0.63|0.71| 315
| 16| 0.75|0.66|0.70| 301
| 17| 0.85|0.90|0.87| 698
| 18| 0.64|0.62|0.63| 584
|avg / total| 0.79|0.79|0.79| 10228
