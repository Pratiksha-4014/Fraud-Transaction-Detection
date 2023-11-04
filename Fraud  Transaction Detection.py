#!/usr/bin/env python
# coding: utf-8

# # Name - Kore Pratiksha Jayant

# # Task 1 - Fraud Transaction Detection

# # 

# # EDA and Visualization

# In[1]:


##Import Library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


##Import Dataset

df=pd.read_csv('C:\\Users\\stati\\OneDrive\\Desktop\\credit card.csv')
df.head()


# In[3]:


##To check null value

df.isnull().sum()
df.Class.unique()


# In[4]:


##To check duplicate value

df.duplicated().sum()


# In[5]:


data=df.drop_duplicates()


# In[6]:


##To check Data types

df.dtypes


# In[7]:


##Data Balance

plt.figure(figsize=[7,3])
plt.subplot(1,2,1)
sns.countplot(x='Class',data=data,color='lime',edgecolor="red")
plt.show()


# This count plot shows that the target class is imbalanced.

# In[8]:


x=data.drop('Class',axis=1);x
y=data['Class'];y


# In[9]:


##To Handel Imbalanced Data

normal=data[data["Class"]==0]
fraud=data[data["Class"]==1]
print(normal.shape)
print(fraud.shape)


# In[10]:


normal_sample=normal.sample(n=473)
normal_sample.shape


# In[11]:


New_data=pd.concat([normal_sample,fraud],ignore_index=True)
New_data


# In[12]:


plt.figure(figsize=[7,3])
plt.subplot(1,2,1)
sns.countplot(x='Class',data=New_data,color='red',edgecolor="black")
plt.show()


# # Summary Statistics

# In[13]:


New_data.describe().style.background_gradient()


# # Visualization

# In[14]:


##Box Plot

plt.figure(figsize=(10, 10))
for i, col in enumerate(New_data.select_dtypes(include=['float64']).columns):
    plt.rcParams['axes.facecolor'] = 'gray'
    ax = plt.subplot(6,5, i+1)
    sns.boxplot(data=New_data, x=col, ax=ax,color='red')
plt.suptitle('Box Plot of continuous variables')
plt.tight_layout()


# In the dataset there is more outliers so we replase the outliers with median from the dataset.

# In[15]:


##Replace the outliers with median from the dataset

def outlier_treating(data,var):
    df=data.copy()#creating a copy of the data
    def outlier_detector(data):                                           #detecting the outliers
        outliers=[]
        q1=np.percentile(data,25)
        q3=np.percentile(data,75)
        IQR=q3-q1
        lb=q1-(IQR*1.5)
        ub=q3+(IQR*1.5)
        for i,j in enumerate(data):
            if(j<lb or j>ub):
                outliers.append(i)
        return outliers
    for i in var:
        out_var=outlier_detector(df[i])                                   #calling outlier_detector function 
        df.loc[out_var,i]=np.median(df[i])                                #replacing the outliers to the median
    return df


# In[16]:


##To handel Outliers

var=list(New_data.select_dtypes(include=['float64']).columns)


# In[17]:


New_df=outlier_treating(New_data,var)


# In[18]:


##After treating outliers

plt.figure(figsize=(18, 18))
for i, col in enumerate(New_df.select_dtypes(include=['float64']).columns):
    plt.rcParams['axes.facecolor'] = 'gray'
    ax = plt.subplot(6,5, i+1)
    sns.boxplot(data=New_df, x=col, ax=ax,color='red')
plt.suptitle('Box Plot of continuous variables')
plt.tight_layout()


# In[19]:


##Correlation Heatmap

df_num=New_df
plt.figure(figsize=(18,18))
sns.heatmap(data=df_num.corr(),cmap="PiYG",annot=True,center=0)


# In[20]:


data['Class'].value_counts()


# # Data Preprocessing

# In[21]:


##Import Library

from sklearn.model_selection import train_test_split


# In[22]:


##Split inputs and output features

x=data.drop('Class',axis=1)
y=data['Class']


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)


# In[24]:


##Train and Test Dataset

from sklearn.preprocessing import StandardScaler


# In[25]:


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# # Model Training and Model Evaluation

# import all library need for model training

# In[26]:


##Importing all library need for model training

##Logistic Regression
from sklearn.linear_model import LogisticRegression

##Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

##Decision Tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.metrics import classification_report


# Logistic Regression

# In[27]:


LR=LogisticRegression(max_iter=500)
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)


# In[28]:


print("Model Accuracy :",accuracy_score(y_pred,y_test))
print("Model F1-Score :",f1_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[29]:


accuracies = cross_val_score(estimator=LR,X=x_train,y=y_train,cv=5)
print("Cross Val Accuracy:",format(accuracies.mean()))
print("Cross Val Standard Deviation:",format(accuracies.std()))


# Decision Tree

# In[30]:


DT=DecisionTreeClassifier(criterion='entropy',random_state=42)
DT.fit(x_train,y_train)
y_pred1=DT.predict(x_test)


# In[31]:


print("Model Accuracy :",accuracy_score(y_pred1,y_test))
print("Model F1-Score :",f1_score(y_pred1,y_test))
print(classification_report(y_pred1,y_test))


# In[32]:


accuracies = cross_val_score(estimator=DT,X=x_train,y=y_train,cv=5)
print("Cross Val Accuracy:",format(accuracies.mean()))
print("Cross Val Standard Deviation:",format(accuracies.std()))


# In[33]:


RF=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=42)
RF.fit(x_train,y_train)
y_pred2=RF.predict(x_test)


# In[34]:


print("Model Accuracy :",accuracy_score(y_pred2,y_test))
print("Model F1-Score :",f1_score(y_pred2,y_test))
print(classification_report(y_pred2,y_test,zero_division=1))


# In[35]:


accuracies = cross_val_score(estimator=RF,X=x_train,y=y_train,cv=5)
print("Cross Val Accuracy:",format(accuracies.mean()))
print("Cross Val Standard Deviation:",format(accuracies.std()))


# In[36]:


#creating dictionary for storing different models accuracy
model_comparison={}


# In[37]:


model_comparison['Logistic Regression']=[accuracy_score(y_pred,y_test),f1_score(y_pred,y_test,average='weighted'),(accuracies.mean()),(accuracies.std())]
model_comparison['Decision Tree']=[accuracy_score(y_pred,y_test),f1_score(y_pred,y_test,average='weighted'),(accuracies.mean()),(accuracies.std())]
model_comparison['Random Forest']=[accuracy_score(y_pred,y_test),f1_score(y_pred,y_test,average='weighted'),(accuracies.mean()),(accuracies.std())]


# In[38]:


Final_df=pd.DataFrame(model_comparison).T
Final_df.columns=['Model Accuracy','Model F1-Score','CV Accuracy','CV std']
Final_df=Final_df.sort_values(by='Model F1-Score',ascending=False)
Final_df.style.format("{:.2%}").background_gradient(cmap='Reds')


# In[41]:


##Over Sampling

X=data.drop('Class',axis=1)
Y=data['Class']


# In[42]:


X.shape


# In[43]:


Y.shape


# In[44]:


get_ipython().system('pip install imblearn')


# In[45]:


from imblearn.over_sampling import SMOTE


# In[46]:


x_res,y_res=SMOTE().fit_resample(X,Y)


# In[47]:


y_res.value_counts()


# In[48]:


X_train,X_test,Y_train,Y_test=train_test_split(x_res,y_res,test_size=0.3,random_state=42)


# In[49]:


def mymodel(model):
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    
    print("Model Accuracy :",accuracy_score(Y_pred,Y_test))
    print("Model F1-Score :",f1_score(Y_pred,Y_test))
    print(classification_report(Y_pred,Y_test,zero_division=1))
    
    return model


# In[50]:


lr=mymodel(LogisticRegression())


# In[51]:


dt=mymodel(DecisionTreeClassifier())


# In[52]:


rf=mymodel(RandomForestClassifier())


# Here we have got the best model by using Random Forest Classifier algoritham.

# In[53]:


RF1=RandomForestClassifier()
RF1.fit(x_res,y_res)


# In[54]:


get_ipython().system('pip install joblib')


# In[55]:


import joblib


# In[56]:


joblib.dump(RF1,"Credit_Card_Model")


# In[57]:


model=joblib.load("Credit_Card_Model")


# In[58]:


##values of v in dataset
prediction=model.predict([[1,2,3,1,2,1,1,4,1,5,1,3,1,1,8,1,7,1,1,2,2,1,5,1,7,1,1,1,1,1]])


# In[59]:


if prediction==0:
    print("Normal Transaction")
else:
    print("Fraud Transaction")

# End