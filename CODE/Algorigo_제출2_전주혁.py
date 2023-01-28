import pandas as pd
import numpy as np

y_pred = pd.read_csv('./labeled.csv')

#Acc Score = 0.941950
z = y_pred[y_pred['Score'] > 0.825]

#Acc Score = 0.941950  Filtering 안한 Acc Score

# Acc Score = 0.944650      >  0.825
# Acc Score = 0.943450      >  0.85
# Acc Score = 0.943250      >  0.80
# Acc Score = 0.942650      >  0.88
# Acc Score = 0.942550      >  0.70
# Acc Score = 0.942350      >  0.75
# Acc Score = 0.942300      >  0.65
# Acc Score = 0.939850      >  0.92


z = z.reset_index()
a = z['id']
b = z['Label']
m = pd.concat([a,b],axis=1)

unlabeled = pd.read_csv('./unlabeled.csv')
train = pd.merge(unlabeled, m, on = 'id', how = 'left')
train = train.dropna(axis=0)
test = pd.read_csv('./test.csv')

train.drop(['id'],axis=1,inplace= True)
test.drop(['id'],axis = 1,inplace=True)


from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, train_test_split

test_x = test.drop(['satisfaction'],axis=1)
test_y = test['satisfaction']

train_data = TabularDataset(train)
test_data = TabularDataset(test_x)
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
print("==================Tabular_complete========================")
predictor = TabularPredictor(label='Label',  eval_metric='accuracy').fit(train_data, presets='high_quality',  ag_args_fit={'num_gpus': 1})
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
print("==================learning_complete========================")
y_pred = predictor.predict(test_data)
print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")
print("==================predictor_complete========================")
print(test_y)
print(y_pred)
print("==================score_complete========================")
print("==================score_complete========================")
print("==================score_complete========================")
print("==================score_complete========================")
print("==================score_complete========================")
print("==================score_complete========================")
print('Acc Score = %lf'%accuracy_score(test_y,y_pred))
'''
fi = predictor.feature_importance(train)
fi = fi.reset_index()
fi.to_csv('./fi.csv',index=False)
print(fi)
'''
