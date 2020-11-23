#!/usr/bin/env python
# coding: utf-8

# In[334]:


import imblearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline as imb_pipline
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from math import sqrt
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss

sns.set()


# In[335]:


data = pd.read_csv('../p_train.csv')
target = pd.read_csv('../PerStatus.csv')
data.drop(['Work Overtime'], inplace=True, axis=1)


# In[336]:


test = pd.read_csv('../E_data/stest.csv')
test.drop(['Unnamed: 0', 'PerStatus', 'PerNo', 'Work Overtime'], axis=1, inplace=True)
test.columns = data.columns
test.ffill(inplace=True)
test.fillna(0, inplace=True)


# In[337]:


full = data.append(test, ignore_index=True)


# In[338]:


X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.25)


# In[339]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

categorical_columns = full.columns
categorical_pipeline = make_pipeline(
    SimpleImputer(missing_values=-1, strategy='most_frequent'),
    OneHotEncoder(categories='auto'))

preprocessor = ColumnTransformer(
    [('categorical_preprocessing', categorical_pipeline, categorical_columns)],
    remainder='drop')


# In[340]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall

def build_model(n_features):
    model = Sequential()
    model.add(Dense(100, input_shape=(n_features,),
              kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_initializer='glorot_normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', Recall(name='recall')])

    return model


# In[341]:


from sklearn.metrics import roc_auc_score
from imblearn.keras import BalancedBatchGenerator

def fit_predict_balanced_model(X_train, Y_train, X_test, Y_test):
    model = build_model(X_train.shape[1])
    training_generator = BalancedBatchGenerator(X_train, Y_train,
                                                batch_size=100,
                                                random_state=42)
    model.fit_generator(generator=training_generator, epochs=50, verbose=1)
    y_pred = model.predict_proba(X_test, batch_size=300)
    return roc_auc_score(Y_test, y_pred), model


# In[342]:


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10)
full = preprocessor.fit_transform(full)
models = []

cv_results_balanced = []
for train_idx, valid_idx in skf.split(X_train, Y_train):
    X_local_train = preprocessor.transform(X_train.iloc[train_idx])
    y_local_train = Y_train.iloc[train_idx].values.ravel()
    X_local_test = preprocessor.transform(X_train.iloc[valid_idx])
    y_local_test = Y_train.iloc[valid_idx].values.ravel()

    roc_auc, model = fit_predict_balanced_model(
        X_local_train, y_local_train, X_local_test, y_local_test)
    models.append(model)
    cv_results_balanced.append(roc_auc)


# In[343]:


df_results = (pd.DataFrame({'Imbalanced model': cv_results_balanced}).unstack().reset_index())

plt.figure()
sns.boxplot(y='level_0', x=0, data=df_results, whis=10.0)
sns.despine(top=True, right=True, left=True)
ax = plt.gca()
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x)))
plt.xlabel('ROC-AUC')
plt.ylabel('')
plt.title('Difference in terms of ROC-AUC using a random under-sampling')


# In[326]:


def store_csv(prediction, filename):
    sub = pd.read_csv('../submission.csv')
    new = {'PerStatus':prediction}
    sub.update(new)
    sub.to_csv(filename, index=False)


# In[327]:


test = preprocessor.transform(test)


# In[328]:


results = models[0].predict(test)
results = results[:, 0]
results


# In[329]:


thresh = 0.8

for i in range(len(results)):
    if results[i] > thresh:
        results[i] = 1
    else:
        results[i] = 0


# In[330]:


results.sum()


# In[331]:


store_csv(results, 'balanced_dnn3.csv')


# # -------------------------------------------------------------

# In[94]:


model = build_model()
model.summary()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', Recall(name='recall')])


# In[95]:


checkpoint = ModelCheckpoint('dnn_selected.h5', monitor='val_accuracy', verbose=1, save_best_only=True, 
                            save_weights_only=False, mode='auto', save_freq=1)

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100, verbose=1, mode='auto')

LR_adj=ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)


# In[96]:


history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=1, 
                    validation_split=0.2, callbacks=[early, LR_adj])


# In[97]:


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()


# In[98]:


show_train_history(history, 'accuracy', 'val_accuracy')
show_train_history(history, 'loss', 'val_loss')


# In[126]:


test = preprocessing.scale(test)
result = model.predict(test)


# In[128]:


thresh = result.mean()

for i in range(len(result)):
    if result[i] > thresh:
        result[i] = 1
    else:
        result[i] = 0


# In[129]:


result.sum()


# In[130]:


store_csv(result[:, 0].astype(int), 'ada_DNN.csv')


# In[ ]:




