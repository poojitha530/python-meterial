#!/usr/bin/env python
# coding: utf-8

# #### Machine learning:
# #### shift from stat to ML technique.
# #### if you are calculate mathematical forms that is in the form of vector.for that we r using natural language .
# #### dependent variables are predict using vector and function we r using for this machine learning function.
# #### steps:
# #### preprocessing:1.data cleaning & standardisation,2.feature engineering this two under the process of conver to vectors.
# #### model building:1.creating modelfor different algorithms,2.checking accuracy,A.data,B.model.again data divided into three types :1.training,2.validation,3.testing.
# #### overfitting:working on trained data & not testing data
# #### underfitting:it's does not have the training data & testing data
# #### middle of the overfitting and underfitting that is known as general fit/generalised model.
# #### for each parameter will get one accuracy.
# #### testing data accuracy is real data accuracy.
# #### data pre processing:
# #### 1.imputation:to remove none value.code: from sqlimputation import simpleimputer
# #### 2. standarisation:converting a data point to standarised data points.
# #### 3.min,max scaler:normalise data point we will use these saclers.
# #### 4.label encoding:converting text values to numerical data points.
# #### 5.one hot encoding:to convert all the data points into pass matrixs(all data points are in 0's and 1's)
# #### 6.model building:1.linear regression,2.logistic regression,3.polynomial features,4.k-nearest neiboures
# #### k-nearest neighbors:
# #### These techniques were used to find the distance between two data points of k-nearest neighbors:closeness or distance calculated methods:1.euclideian distance(L2 normalisation) 2.man hatten distance(L1 normalisation) 3.minkowski 4.hamming distance(dissimilarv alues =distance) 5.cosine distance or cosine similarity(angular based distance)(similar is inversly propotional to 1/distace)
# #### 1.euclideian distance:c=(a^2+b^2)^2 -->L2 normalisation
# #### pipe line building:model will build fastly while using pipe line 
# #### validation techniques:
# #### k-fold cross validation:data is divided into different types of folds or data optimization is known as k-fold cross validation.
# #### grid seach cross validation:parameter optimisation technique
# #### randam search cross validation:parameter optimisation 
# #### model.predict()-->accuracy
# #### -->on test data-->metrics-->confusion matrix:TP(true positive),FP(false positive),TN(true negative),FN based on these 1.accuracy,2.precision,3.recall,4.Fscore,5.Roc Curve
# #### Deployment:1.saving model with extension (.sav) 2.Pickel(file-->open,write),joblib(direct dump) these two libraries are used to dump model & load it in other file.
# #### -->other file(model predict(data point))+interface
# #### descriminant analyasis we used in PCA
# 
# 

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# In[2]:


df1=pd.read_csv("IRIS.csv")


# In[3]:


from sklearn.impute import SimpleImputer


# In[4]:


#splitting the train and test data set
from sklearn.model_selection import train_test_split as tt
df=pd.read_csv("Data.csv")
X=df.iloc[:,:-1].values # x is independent variables
Y=df.iloc[:,-1].values # y is dependent varaibles
xtrain,xtest,ytrain,ytest=tt(X,Y,test_size=0.3)
ytest


# In[5]:


#splitting the train and test data set
from sklearn.model_selection import train_test_split as tt
df=pd.read_csv("Data.csv")
X=df.iloc[:,:-1].values # x is independent variables
Y=df.iloc[:,-1].values # y is dependent varaibles
xtrain,xtest,ytrain,ytest=tt(X,Y,test_size=0.3)
xtest


# In[6]:


#splitting the train and test data set
from sklearn.model_selection import train_test_split as tt
df=pd.read_csv("Data.csv")
X=df.iloc[:,:-1].values # x is independent variables
Y=df.iloc[:,-1].values # y is dependent varaibles
xtrain,xtest,ytrain,ytest=tt(X,Y,test_size=0.3)
xtrain


# In[7]:


#splitting the train and test data set
from sklearn.model_selection import train_test_split as tt
df=pd.read_csv("Data.csv")
X=df.iloc[:,:-1].values # x is independent variables
Y=df.iloc[:,-1].values # y is dependent varaibles
xtrain,xtest,ytrain,ytest=tt(X,Y,test_size=0.3)
ytrain


# In[8]:


#splitting the train and test data set
from sklearn.model_selection import train_test_split as tt

xtrain,xtest,ytrain,ytest=tt(X,Y,test_size=0.3)
xtest


# In[9]:


df1.info



# In[10]:


df.info


# In[11]:


df=pd.read_csv("Data.csv")
X=df.iloc[:,:-1].values # x is independent variables
Y=df.iloc[:,3].values
df.value_counts()


# In[12]:


df=pd.read_csv("Data.csv")
X=df.iloc[:,:-1].values # x is independent variables
Y=df.iloc[:,3].values
df


# In[13]:


from sklearn.preprocessing import LabelEncoder as le
l=le()
Y=l.fit_transform(Y)
Y


# In[14]:


from sklearn.preprocessing import LabelEncoder as le
l=le();X[:,0]=l.fit_transform(X[:,0])
Y=l.inverse_transform(Y)
Y


# In[15]:


from sklearn.preprocessing import OneHotEncoder as oc
o=oc()
X=o.fit_transform(X).toarray()
X


# In[16]:


#feature scaling :standardisation
from sklearn.preprocessing import StandardScaler as sc
s=sc()
X=s.fit_transform(X)


# In[17]:


#splitting the train and test dataset
from sklearn.model_selection import train_test_split as tt
xtrain,xtest,ytrain,ytest=tt(X,Y,test_size=0.3)
xtest


# In[18]:


from sklearn.neighbors import NearestNeighbors as knn


# In[19]:


kn=knn()


# In[20]:


model=kn.fit(xtrain,ytrain)
model.kneighbors_graph(xtest).toarray()


# ### afternoon session 

# In[26]:


sn=pd.read_csv("Social_Network_Ads.csv")


# In[28]:


x=sn.iloc[:,[2,3]].values
y=sn.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
x_test


# In[44]:


x=sn.iloc[:,[2,3]].values
y=sn.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#fitting kernel SVM to the training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred).ravel()
#applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X =x_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()




# In[45]:


#applying grid search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
            {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1) 
grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_


# In[46]:


print(best_accuracy*100)


# In[47]:


print(best_parameters)


# In[48]:


print(y_pred)


# In[49]:


print(cm)


# In[50]:


tn,fp,fn,tp=cm


# In[51]:


tp


# In[52]:


fp


# In[53]:


fn


# In[54]:


tn


# In[57]:


import joblib
joblib.dump(classifier,'SVC_classifier.sav')


# In[62]:


import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler as sc


# In[63]:


model=joblib.load("SVC_classifier.sav")


# In[73]:


parameters=np.array([26,43000])

model.predict([parameters])


# ## Decision tree classification algorithm

# In[74]:


from sklearn.tree import DecisionTreeClassifier as Classifier


# In[101]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as l,OneHotEncoder as o, StandardScaler as s
df=pd.read_csv("Social_Network_Ads.csv")
x=df.iloc[:,1:-1]
y=df.iloc[:,-1]
le=l()
x.iloc[:,0]=le.fit_transform(x.iloc[:,0])
std=s()
x=std.fit_transform(x)
oe=o()
x=std.fit_transform(x)



# In[108]:


from sklearn.model_selection import train_test_split as tp
xtrain,xtest,ytrain,ytest=tp(x,y,test_size=0.4)


# In[110]:


from sklearn.tree import DecisionTreeClassifier as classifier
C=classifier()
model=C.fit(xtrain,ytrain)
model.predict(xtest)


# In[ ]:




