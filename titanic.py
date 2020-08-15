# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:01:48 2020

@author: patno_000
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('datacamp_projects/titanic.csv')

df.head()
cols = list(df.columns)
cols = cols[0:2] + cols[3:] #remove name because about to get dummies

df1 = df[cols]
df1.head()
df1.describe()
df1.dtypes

df2 = pd.get_dummies(df1, drop_first = True)
df2.head()
df2.describe()
df2.dtypes

X = df2.iloc[:,1:]
y = df2.iloc[:,0]

from sklearn.linear_model import LogisticRegression
logreg1 = LogisticRegression()

logreg1.fit(X,y)
from sklearn.model_selection import cross_val_score
cv = cross_val_score(logreg1, X, y, cv=5)

print(cv)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=41)

logreg1.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, classification_report

y_pred = logreg1.predict(X_test) #predict 0,1



# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


###ROC Curve
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg1.predict_proba(X_test)[:,1] #predict probability of being 1

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

liv = df2[df2['Survived']==1]
die = df2[df2['Survived']!=1]
for col in X.columns:
    plt.hist(liv[col], color='white', alpha=0.3, edgecolor='blue', linewidth=4)
    plt.hist(die[col], color='white', alpha=0.3, edgecolor='green', linewidth=4)
    #liv.plot(col,  kind="density")
    plt.show()
    print(col)

X.columns
import numpy as np
##### Predict with new data
PM_test = [[1, 27, 0, 0, 25, 1],
 [2, 27, 0, 0, 25, 1],
 [3, 27, 0, 0, 25, 1],
 [1, 27, 0, 0, 50, 1],
 [2, 27, 0, 0, 50, 1],
 [3, 27, 0, 0, 50, 1],
 [1, 27, 1, 0, 25, 1],
 [2, 27, 1, 0, 25, 1],
 [3, 27, 1, 0, 25, 1],
 [1, 27, 0, 0, 25, 0],
 [2, 27, 0, 0, 25, 0],
 [3, 27, 0, 0, 25, 0],
 [1, 27, 0, 0, 50, 0],
 [2, 27, 0, 0, 50, 0],
 [3, 27, 0, 0, 50, 0],
 [1, 27, 1, 0, 25, 0],
 [2, 27, 1, 0, 25, 0],
 [3, 27, 1, 0, 25, 0]]

PM_test_ar = np.array(PM_test)
PM_test_df = pd.DataFrame(PM_test_ar)
PM_test_df.columns = X.columns    
logreg1.fit(X,y)
logreg1.predict(PM_test_df)
logreg1.predict_proba(PM_test_df)


from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

NB = GaussianNB()
NB.fit(X_train, y_train)

NB.score(X_test, y_test)

NB.predict(X_test)

NB.predict(PM_test_df)

