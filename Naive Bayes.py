#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


salary = pd.read_csv('F:/Dataset/Salary_Data.csv')


# In[4]:


salary


# In[5]:


salarytrain = pd.read_csv('F:/Dataset/SalaryData_Train(1).csv')


# In[6]:


salarytrain


# In[7]:


salarytest = pd.read_csv("F:/Dataset/SalaryData_Test(1).csv")


# In[9]:


salarytest


# In[11]:


salarytrain.info()


# In[12]:


salarytest.info()


# In[13]:


salarytest.describe()


# In[14]:


salarytrain.describe()


# In[15]:


salarytrain.isin(['?']).sum(axis=0)


# In[17]:


salarytest.isin(['?']).sum(axis=0)


# In[18]:


print(salarytrain[0:5])


# In[19]:


print(salarytest[0:5])


# In[20]:


categorical = [var for var in salarytrain.columns if salarytrain[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)


# In[22]:


salarytrain[categorical].head()


# In[23]:


salarytrain[categorical].isnull().sum()


# In[24]:


for var in categorical: 
    
    print(salarytrain[var].value_counts())


# In[25]:


for var in categorical: 
    
    print(salarytrain[var].value_counts()/np.float(len(salarytrain)))


# In[27]:


salarytrain.workclass.unique()


# In[28]:


salarytrain.workclass.value_counts()


# In[29]:


salarytrain.occupation.unique()


# In[30]:


salarytrain.occupation.value_counts()


# In[31]:


salarytrain.native.unique()


# In[32]:


salarytrain.native.value_counts()


# In[33]:


for var in categorical:
    
    print(var, ' contains ', len(salarytrain[var].unique()), ' labels')


# In[34]:


numerical = [var for var in salarytrain.columns if salarytrain[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[35]:


salarytrain[numerical].head()


# In[36]:


salarytrain[numerical].isnull().sum()


# In[37]:


X = salarytrain.drop(['Salary'], axis=1)

y = salarytrain['Salary']


# In[38]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[39]:


X_train.shape, X_test.shape


# In[40]:


X_train.dtypes


# In[41]:


X_test.dtypes


# In[42]:


categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[43]:


numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[44]:


X_train[categorical].isnull().mean()


# In[45]:


for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))


# In[46]:


for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native'].fillna(X_train['native'].mode()[0], inplace=True)


# In[47]:


X_train[categorical].isnull().sum()


# In[48]:


X_test[categorical].isnull().sum()


# In[49]:


X_train.isnull().sum()


# In[50]:


X_test.isnull().sum()


# In[51]:


categorical


# In[52]:


X_train[categorical].head()


# In[54]:


get_ipython().system('pip install category_encoders')


# In[55]:


import category_encoders as ce


# In[56]:


encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[57]:


X_train.head()


# In[58]:


X_train.shape


# In[59]:


X_test.shape


# In[60]:


cols = X_train.columns


# In[61]:


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[63]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[64]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[65]:


X_train


# In[66]:


from sklearn.naive_bayes import GaussianNB


# In[67]:


gnb = GaussianNB()


# In[70]:


gnb.fit(X_train, y_train)


# In[71]:


y_pred = gnb.predict(X_test)


# In[72]:


y_pred


# In[73]:


from sklearn.metrics import accuracy_score


# In[74]:


print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[75]:


y_pred_train = gnb.predict(X_train)


# In[76]:


y_pred_train


# In[77]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[78]:


print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))


# In[79]:


y_test.value_counts()


# In[80]:


null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# In[81]:


from sklearn.metrics import confusion_matrix


# In[82]:


cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[83]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[84]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[85]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[86]:


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# In[87]:


classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))


# In[88]:


precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))


# In[89]:


recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# In[90]:


true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# In[91]:


false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# In[92]:


specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))


# In[93]:


y_pred_prob = gnb.predict_proba(X_test)[0:10]
y_pred_prob


# In[94]:


y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])
y_pred_prob_df


# In[95]:


gnb.predict_proba(X_test)[0:10, 1]


# In[96]:


y_pred1 = gnb.predict_proba(X_test)[:, 1]


# In[97]:


plt.hist(y_pred1, bins = 10)


# In[98]:


from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_test, y_pred1)
print('ROC AUC : {:.4f}'.format(ROC_AUC))


# In[99]:


from sklearn.model_selection import cross_val_score
Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


# In[100]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))


# In[101]:


print('Average cross-validation score: {:.4f}'.format(scores.mean()))


# In[ ]:




