#j

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import json

#loading dataset
f = open('newdataset.json','r') #read json from the file
d = json.load(f) #read from file 
df=pd.DataFrame(d)
df['RFA_2A']=df['RFA_2A'].map({'E':1, 'D':2, 'G':3, 'F':4}) #convert to numeric
del df['PEPSTRFL']
del df['NAME']

#balancing the dataset
respond=df.loc[df['TARGET_B']==1]
nonrespond=df.loc[df['TARGET_B']==0]
sampled0s=nonrespond.sample(len(respond))
balanced=respond.append(sampled0s)
balFeatures=balanced.drop(['TARGET_B'],axis=1)
balTarget=balanced['TARGET_B']

#splitting train and test data set

X_train, X_test, Y_train, Y_test = train_test_split(balFeatures, balTarget, test_size=0.2, random_state=42)

#normalizing X

min_max_scaler=preprocessing.MinMaxScaler()
X_train_scaled=min_max_scaler.fit_transform(X_train)
df_X_train_scaled=pd.DataFrame(X_train_scaled)
X_test_scaled=min_max_scaler.fit_transform(X_test)
df_X_test_scaled=pd.DataFrame(X_test_scaled)

#random forest classifier

clf = RandomForestClassifier(n_estimators=10)
clf.fit(df_X_train_scaled, Y_train)
predicted=clf.predict(df_X_test_scaled)

#plotting ROC & AUC
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, predicted)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
	
# measuring performance score
accuracy_score(Y_test,clf.predict(df_X_test_scaled))


#logistic regression

clf = linear_model.LogisticRegression()
trained=clf.fit(df_X_train_scaled,Y_train)
predicted=clf.predict(df_X_test_scaled)


#plotting ROC & AUC
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, predicted)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#measuring performance score
accuracy_score(Y_test,clf.predict(df_X_test_scaled))


#tree classifier

clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(max_depth=2)
trained=clf.fit(df_X_train_scaled,Y_train)
predicted=clf.predict(df_X_test_scaled)

#plotting ROC & AUC
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, predicted)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#measuring performance score
accuracy_score(Y_test,clf.predict(df_X_test_scaled)) 

		
