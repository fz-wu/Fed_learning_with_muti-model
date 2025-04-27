import pandas as pd
from sklearn.datasets import load_breast_cancer
import pickle
data=load_breast_cancer()
dataset=pd.DataFrame(data=data['data'],columns=data['feature_names'])
from sklearn.model_selection import train_test_split
x=dataset.copy()
y=data['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.01, random_state=42)
clf=clf.fit(x_train,y_train)

predictions=clf.predict(x_test)

pickle.dump(clf, open('fed_dt.pkl', 'wb'))