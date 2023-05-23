# -*- coding: utf-8 -*-
"""
Heart Disease Project
Created on Tue May 23 12:13:43 2023
@author: Sourav
"""
# Import the library and dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data= pd.read_csv("heart.csv")

data.head(10).transpose()
data.columns
data.describe().transpose()
data.shape ## 1025 rows and 14 columns
data.info()


## Checking null values in the dataset
data.isnull().sum()
# we donot have any null values in our dataset

## Check duplicate data values
da_dup= data.duplicated().any()
print(da_dup)
data= data.drop_duplicates()

data.shape ## 302 rows and 14 columns

## Drawing correlation matrix
plt.figure(figsize=(17,10),dpi=200)
sns.heatmap(data.corr(),annot=True)

## Having or not having heart disease
data['target'].value_counts() ## 1- 164 and 0- 138
sns.countplot(data['target'])

## Count og M and F
data['sex'].value_counts() ## 1- 206 and 0- 96
sns.countplot(data['sex'])
plt.xticks([0,1],['Female', 'Male'])

# Find general distribution acc to target variable
sns.countplot(data['sex'], hue=data['target'])
plt.xticks([0,1],['Female', 'Male'])

## Age distribution of the datset
sns.distplot(data['age'],bins=15)

## Chest pain type

sns.countplot(data['cp'])
plt.xticks([0,1,2,3],['ta', 'aa','nap','as'])

sns.countplot(x='cp', hue='target', data= data)
plt.xticks([0,1,2,3],['ta', 'aa','nap','as'])

sns.countplot(x='fbs', hue='target', data= data)
plt.legend(labels= ['No-Disease', 'Disease'])

data['trestbps'].hist()

## Resting blood pressure vs sex column

g= sns.FacetGrid(data, hue= 'sex', aspect=4)
g.map(sns.kdeplot,'trestbps', shade= True)
plt.legend(labels= ['Male', 'Female'])

# Show the distribution of Serum Cholestrol
data['chol'].hist()

# plot continuous variables
data.columns

## Seperating categorical and numerical columns
cate_val=[]
cont_val=[]
for column in data.columns:
    if data[column].nunique()<=10:
        cate_val.append(column)
    else:
        cont_val.append(column)

cate_val ## ['sex','cp','fbs','restecg','exang','slope','ca','thal','target']
cont_val  ## ['age','trestbps','chol','thalach','oldpeak']

data.hist(cont_val, figsize= (15,10))

data.isnull().sum() # no missing values





######################### Preprocessing #######################

cate_val.remove('sex')
cate_val.remove('target')
data= pd.get_dummies(data, columns= cate_val, drop_first=True)
data.head().transpose() ## 23 columns after the dummies created

## Feature scaling
from sklearn.preprocessing import StandardScaler
st= StandardScaler()
data[cont_val]= st.fit_transform(data[cont_val])


## Splitting the data

X= data.drop('target', axis=1)
y= data['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




################################ Modelling #############################

## We have a classification problem on our disposal



## 1- Logistic Regression

from sklearn.linear_model import LogisticRegression
log= LogisticRegression()
log.fit(X_train,y_train)

y_pred1= log.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred1)   ## 78.68 % accurate




## 2 SVC support vector Classifier

from sklearn import svm
svm= svm.SVC()
svm.fit(X_train, y_train)

y_pred2= svm.predict(X_test)
accuracy_score(y_test, y_pred2) ## 80.32 % accurate

## 3 KNeighbors Classifier

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred3= knn.predict(X_test)
accuracy_score(y_test, y_pred3) ## 73.77% accurate

score=[]
for i in range(1,40):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred= knn.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))

score  ## Best 81% accuracy




####################### Non Linear ML Algorithms ############
## No need of feature scaling


df= pd.read_csv('heart.csv')
df= df.drop_duplicates()
df.shape

X= df.drop('target', axis=1)
y= df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




## 4 Decision Tree Clasifier
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred4= dt.predict(X_test)
accuracy_score(y_test, y_pred4)  ## 77% accurate




## 5 Randomforest Classifier
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(X_train, y_train) 

y_pred5= rf.predict(X_test)
accuracy_score(y_test,y_pred5) ## 83% accurate


## 6 Gradient boosting

from sklearn.ensemble import GradientBoostingClassifier
gd= GradientBoostingClassifier()
gd.fit(X_train,y_train)

y_pred6= gd.predict(X_test)
accuracy_score(y_test,y_pred6) ## 80% accurate



################ Final Interpretation ###############

final_data= pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],
                          'ACC':[accuracy_score(y_test,y_pred1),
                                 accuracy_score(y_test,y_pred2),
                                 accuracy_score(y_test,y_pred3),
                                 accuracy_score(y_test,y_pred4),
                                 accuracy_score(y_test,y_pred5),
                                 accuracy_score(y_test,y_pred6)]})
final_data
sns.barplot(final_data['Models'],final_data['ACC'])



## 5 Randomforest Classifier
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(X, y)

#3 Saving model using Joblib
import joblib
joblib.dump(rf,'model_joblib_heart')


## Loading the model
model= joblib.load('model_joblib_heart')











