#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('data/train.csv')


# ### Exploratory Data Analysis
# Data Exploration and Cleaning - Adesh

# In[3]:


train.head(5)


# Feature Breakdown:
# 
# - Categorical: Survived, Pclass, Sex, Embarked
# - Numerical: Age, Fare, SibSp, Parch
# - Mixed: Name, Ticket, Cabin

# In[4]:


train.describe()


# In[5]:


sns.heatmap(train.isnull(), cbar=False)


# Assumptions:
# - `passengerId` has no influence on the data -> drop
# - `Cabin` is missing most of its values and therefore cannot be imputed -> drop
# - `Ticket` is alphanumerical (difficult to work with) and 23% of values aren't unique -> drop?
# - `Name` is a mixture of names and titles, seemingly doesn't affect survival -> drop?

# In[6]:


train['Age'].hist(bins=30)


# In[7]:


sns.countplot(x='Survived', data=train, hue='Sex')


# In[8]:


sns.countplot(x='Survived', data=train, hue='Embarked')


# In[9]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Pclass', y='Age', hue='Sex', data=train)


# In[10]:


medians = train.groupby(['Pclass','Sex'])['Age'].median().values
medians = [str(np.round(s, 2)) for s in medians]
medians


#  - `age_imputation` imputes the age of a passenger with the median value for their respective Pclass and Sex.
#  - `gender` hot encodes male (1) / female (0).
#  - `southampton` and `queenstown` hot encodes the port of embarking.

# In[11]:


def age_imputation (cols):
    Age = cols[0]
    Sex = cols[1]
    Pclass = cols[2]
    
    if pd.isnull(Age):
        if Pclass == 1 and Sex == 'male':
            return 40
        elif Pclass == 1 and Sex == 'female':
            return 35
        elif Pclass == 2 and Sex == 'male':
            return 30
        elif Pclass == 2 and Sex == 'female':
            return 28
        elif Pclass == 3 and Sex == 'male':
            return 25
        elif Pclass == 3 and Sex == 'female':
            return 22
    else:
        return Age

def gender (val):
    if val == 'male':
        return 1
    else:
        return 0

def southampton (col):
    Embarked = col
    
    if Embarked == 'S':
        return 1
    else:
        return 0

def queenstown (col):
    Embarked = col
    
    if Embarked == 'Q':
        return 1
    else:
        return 0
    
def title_cat (x):
    if x == 'Mr':
        return 1
    elif x == 'Miss' or x == 'Mrs' or x == 'Ms' or x == 'Mme' or x == 'Lady' or x == 'Mlle':
        return 2
    else:
        return 3


# In[12]:


train['Age'] = train[['Age','Sex','Pclass']].apply(age_imputation,axis=1)
train['Sex'] = train['Sex'].apply(lambda x: gender(x))

train['S'] = train['Embarked'].apply(lambda x: southampton(x))
train['Q'] = train['Embarked'].apply(lambda x: queenstown(x))

train['Title'] = train['Name'].apply(lambda x: x.split(', ')[1].split('.')[0].strip())
train['Title'] = train['Title'].apply(lambda x: title_cat(x))


# In[13]:


train.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)


# In[14]:


train.head(10)


# In[15]:


sns.heatmap(train.corr(), annot= True)


# In[16]:


train.to_csv('data/train_clean.csv')


# ### Machine Learning Models
# 
# 1. Logistic Regression - Tiago
# 2. Random Forest Classification - Adesh
# 3. Support Vector Machine - Raman
# 4. Neural Network - Adesh
# ___

# **Neural Network using Keras - Adesh**

# Splitting data into `X_train` and `y_train`, scaling the data, and using Keras for Neural Network.

# In[17]:


from sklearn.preprocessing import StandardScaler

X_train = train.iloc[:, 1:10]
y_train = train.iloc[:, 0]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)


# In[18]:


from keras import Sequential
from keras.layers import Dense

classifier = Sequential()

# Creating Hidden layers
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=9))
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

# Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# Compiling the Neural Network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

# Fitting the data to the training dataset
classifier.fit(X_train, y_train, batch_size=128, epochs=256)

# Evaluating Model
eval_model=classifier.evaluate(X_train, y_train)
eval_model


# In[19]:


# Plotting Training Metrics
plt.plot(classifier.history.history['acc'])
plt.plot(classifier.history.history['loss'])
plt.title('Training Metrics')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()


# Importing testing set and making it look like the training set.

# In[20]:


X_test = pd.read_csv('data/test.csv')
X_test['Age'] = X_test[['Age','Sex','Pclass']].apply(age_imputation,axis=1)
X_test['Sex'] = X_test['Sex'].apply(lambda x: gender(x))
X_test['S'] = X_test['Embarked'].apply(lambda x: southampton(x))
X_test['Q'] = X_test['Embarked'].apply(lambda x: queenstown(x))
X_test['Title'] = X_test['Name'].apply(lambda x: x.split(', ')[1].split('.')[0].strip())
X_test['Title'] = X_test['Title'].apply(lambda x: title_cat(x))
X_test.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)

y_test = pd.read_csv('data/gender_submission.csv')
y_test.drop(['PassengerId'],axis=1,inplace=True)


# Using model to predict values.

# In[21]:


y_pred_nn = classifier.predict(X_test)
y_pred_nn = y_pred_nn > 0.5


# Confusion matrix and ROC curve for model accuracy.

# In[22]:


from sklearn.metrics import confusion_matrix
from plotting import plot_confusion_matrix

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

cm_nn = confusion_matrix(y_test, y_pred_nn)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_nn)

plot_confusion_matrix(cm_nn, ['Died', 'Survived'])


# In[23]:


plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Neural Network')
plt.show()

roc_auc_score(y_test, y_pred_nn)


# ___
# **Random Forest Classification - Adesh**

# In[24]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0, n_jobs=-1)
model = rfc.fit(X_train, y_train)


# In[25]:


y_pred_rfc = model.predict(np.nan_to_num(X_test))


# In[26]:


cmrfc = confusion_matrix(y_test, y_pred_rfc)
fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y_test, y_pred_rfc)

plot_confusion_matrix(cmrfc, ['Died', 'Survived'])


# In[27]:


plt.plot(fpr_rfc,tpr_rfc)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Random Forest Classifier')
plt.show()

roc_auc_score(y_test, y_pred_rfc)


# ___

# In[ ]:




