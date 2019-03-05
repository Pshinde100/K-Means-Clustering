# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:59:52 2019

@author: Pranit
"""

# Ch: k-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

path="D:\\Imarticus\\Python\\iris.csv"
iris=pd.read_csv(path)
iris.head(10)

iris.shape

iris=iris.drop(["species"],axis=1)

iris.head()

# Build the k-means model with n cluster
km_model = KMeans(n_clusters = 3).fit(iris)
km_model

# Cluster the dataset
labels=km_model.predict(iris)
labels

# centrodis=kmeans.cluster_centers_
# centroids

# assign cluster to each record
iris['cluster']=labels
iris[0:11]


# count of each cluster
c1 =  len(iris[iris.cluster == 0])
c2 =  len(iris[iris.cluster == 1])
c3 =  len(iris[iris.cluster == 2])


print(c1, c2, c3)

c1 = iris


c1=iris[iris.cluster==0].iloc[0:4]
c2=iris[iris.cluster==1].iloc[0:4]
c3=iris[iris.cluster==2].iloc[0:4]

c1
# total count
iris.cluster.count()

# group by
iris.groupby('cluster').count()

iris.columns

# plot the cluster
clusters = km_model.labels_

def plotclusters(x,y,xlbl,ylbl): 
     plt.scatter(x,y,c=clusters.astype(float))
     plt.xlabel(xlbl)
     plt.ylabel(ylbl)
     plt.title('k-Means')
     plt.show()
     
X=iris.values[:,0] ; Y=iris.values[:,1]
x_label="sepal length"; y_label="sepal width"
plotclusters(X,Y,x_label,y_label)

X1=iris.values[:,2] ; Y1=iris.values[:,3]
x1_label="petal length"; y1_label="petal width"
plotclusters(X1,Y1,x1_label,y1_label)