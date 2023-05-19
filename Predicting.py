
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

df=pd.read_csv("bank-full.csv")

df=df.iloc[:,[16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
#############################Data Exploration#################################

df.head() # To show first 5 rows of the data set

df.columns# To show the number of columns in the data set

df.tail()# To show last 5 rows of the data set

df.isnull().sum() #To check is their any null values in the features of the data set

df.shape # #To check number of observations(rows and columns)

df.dtypes # To check the data types of the features in data set

df.drop_duplicates(inplace=True)#Removing the duplicate values


df.info()#Give information about any null values/missing value

df.describe()#Give statistical information about the data set such as mean ,median ,mode,max value,min value

df.Target.value_counts()#To chech the count of target values

cormat=df.corr()#Helps to understand correlation between the dependent and independent variables

#################################CAT boost algorithm########################################################

from catboost import CatBoostClassifier

model8=CatBoostClassifier()
model8.fit(X_train,y_train)

y_pred8=model8.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred8))
print(confusion_matrix(y_test,y_pred8))
print(classification_report(y_test,y_pred8))


n_errors_CB=print((y_pred8!=y_test).sum())
cohen_kappa_score(y_test,y_pred8) 
print(accuracy_score(y_train,model8.predict(X_train)))

#From all the above model results we can conclude that there is class imbalance probelm.
#Since we can observe training accuracy is more that testing accuracy , precision value
#is more for no class and recall value is less.We can observe majority class for no class and monority class for yes class
#So we need to solve class imbalance problem with the help of SMOTE.

from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=444)
X_train_res,y_train_res=sm.fit_resample(X_train,y_train)


X_train_res.shape
y_train_res.shape
X_test.shape
y_test.shape




















































































