#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np


# In[79]:


data = pd.read_csv("driver-data.csv", index_col="id")
data.head()


# In[80]:


data['profit'] = np.random.randint(6,20, size=len(data))


# In[82]:


x = data.iloc[:, [0,1,2]].values
print(x)


# In[83]:


from sklearn.cluster import KMeans


# In[84]:


#To check optimum number of clusters
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()


# In[85]:


kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(x)


# In[87]:


unique, counts = np.unique(labels, return_counts=True)

dict_data = dict(zip(unique, counts))
dict_data


# In[89]:


data["cluster"] = labels
print(data)


# In[61]:


import seaborn as sns
sns.lmplot('mean_dist_day', 'mean_over_speed_perc', data=data, hue='cluster', palette='coolwarm', size=6, aspect=1, fit_reg=False)


# In[103]:


#For normalising
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler

data2 = data
# print(data2)

scaler = StandardScaler()
scaler.fit(data2.drop('cluster',axis=1))
scaled_features = scaler.transform(data2.drop('cluster',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=data2.columns[:-1])
df_feat.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,data2['cluster'],test_size=0.30)


# In[99]:


#To check optimum number of k neighbours
from sklearn.model_selection import cross_val_score
accuracy_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,data2['cluster'],cv=10)
    accuracy_rate.append(score.mean())
    
plt.figure(figsize=(10,6))

plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')


# In[100]:


#As from above plot we can see the accuracy rate is stable when k=17
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)


# In[101]:


#Classification Report
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

