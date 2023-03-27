import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import seaborn as sns

import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

import pickle 

import warnings
warnings.filterwarnings('ignore')

st.write(''' # Academic Success Prediction App''')

st.sidebar.header('User Input Parameters')

def user_input_features():
  Nationality = st.sidebar.slider('Nationality', 4.3, 7.9, 5.4)
  Displaced  = st.sidebar.slider('Displaced', 2.0, 4.4, 3.4)
  Tuition fees up to date = st.sidebar.slider('Tuition fees up to date', 1.0, 6.9, 1.3)
  Scholarship holder = st.sidebar.slider('Scholarship holder', 0.1, 2.5, 0.2)
  Curricular units 1st sem (approved)=st.sidebar.slider('Curricular units 1st sem (approved)', 4.3, 7.9, 5.4)
  Curricular units 1st sem (grade)=st.sidebar.slider('Curricular units 1st sem (grade)', 4.3, 7.9, 5.4)
  Curricular units 2nd sem (approved)=st.sidebar.slider('Curricular units 2nd sem (approved)', 4.3, 7.9, 5.4)
  Curricular units 2nd sem (grade)=st.sidebar.slider('Curricular units 2nd sem (grade)', 4.3, 7.9, 5.4)

  user_input_data = {'Nationality': Nationality,
               'Displaced': Displaced,
               'Tuition fees up to date': Tuition fees up to date,
               'Scholarship holder': petal_width,
               'Curricular units 1st sem (approved)'=Curricular units 1st sem (approved),
               'Curricular units 1st sem (grade)'=Curricular units 1st sem (grade),
               'Curricular units 2nd sem (approved)'=Curricular units 2nd sem (approved),
               'Curricular units 2nd sem (grade)'=Curricular units 2nd sem (grade)}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

student = pd.read_csv('dataset.csv')
student.shape
student.columns
student.sample(4)
print(student.isnull().sum())
print(student.duplicated().sum())
student['Target'].unique()
student['Target'] = student['Target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})
student.describe()
student.corr()['Target']
fig = px.imshow(student)
fig.show()
student_df = student.iloc[:,[1,11,13,14,15,16,17,20,22,23,26,28,29,34]]
sns.heatmap(student_df)
student_df['Target'].value_counts()
x = student_df['Target'].value_counts().index
y = student_df['Target'].value_counts().values

df = pd.DataFrame({
    'Target': x,
    'Count_T' : y
})

fig = px.pie(df,
             names ='Target', 
             values ='Count_T',
            title='How many dropouts, enrolled & graduates are there in Target column')

fig.update_traces(labels=['Graduate','Dropout','Enrolled'], hole=0.4,textinfo='value+label', pull=[0,0.2,0.1])
fig.show()
fig = px.scatter(student_df, 
             x = 'Curricular units 1st sem (approved)',
             y = 'Curricular units 2nd sem (approved)',
             color = 'Target')
fig.show()
X = student_df.iloc[:,0:13]
y = student_df.iloc[:,-1]
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=10, random_state=0)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)

print("With CV: ",scores.mean())
print("Precision Score: ", precision_score(y_test, y_pred,average='macro'))
print("Recall Score: ", recall_score(y_test, y_pred,average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred,average='macro'))
