import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
df = pd.read_csv("data.csv")
df.describe().T
sns.countplot(data=df, x='private')
plt.figure(figsize=(12,8))
plt.title('Enrollment Statistics in Private vs Public Universities')
sns.histplot(data=df, x="enroll", hue="private", multiple="dodge", shrink=.9)
plt.figure(figsize=(12,8))
plt.title('Alumni Statistics in Private vs Public Universities')
sns.histplot(data=df, x="perc_alumni", hue="private", multiple="stack")
sns.set_style('darkgrid')
#sns.plot('room_board','grad_rate',data=df, hue='private',
           #palette='coolwarm',size=6,aspect=1,fit_reg=False)
sns.set_style('darkgrid')
sns.color_palette("magma", as_cmap=True)
#g = sns.FacetGrid(df,hue="private",size=6,aspect=2)
#g = g.map(plt.hist,'outstate',bins=20,alpha=0.7)
sns.histplot(data=df, x="grad_rate")
sns.set_style('darkgrid')
#g = sns.FacetGrid(df,hue="private",palette='coolwarm',size=6,aspect=2)
#g = g.map(plt.hist,'grad_rate',bins=20,alpha=0.7)
df['encoded'] = df['private'].map({'Yes':1,'No':0})
features_mean =list(df.columns[1:18])
# split dataframe into two based on diagnosis
df_Private=df[df['encoded'] ==1]
df_Public =df[df['encoded'] ==0]
plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(13,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    ax.hist([df_Private[features_mean[idx]],df_Public[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True,  label=['Private','Public'],color=['b','y'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop(['private', 'encoded'], axis=1))
kmeans.cluster_centers_
df['encoded'] = df['private'].map({'Yes':1,'No':0})
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['encoded'],kmeans.labels_))
print(classification_report(df['encoded'],kmeans.labels_))
