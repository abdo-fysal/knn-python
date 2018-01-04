# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:14:41 2017

@author: aboda
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
xblue=np.array([0.3,0.5,1,1.4,1.7,2])
yblue=np.array([1,4.5,2.3,1.9,8.9,4.1])

xred=np.array([3.3,3.5,4,4.4,5.7,6])
yred=np.array([7,1.5,6.3,1.9,2.9,7.1])


x=np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
y=np.array([0,0,0,0,0,0,1,1,1,1,1,1])


plt.plot(xblue,yblue,'ro',color='blue')
plt.plot(xred,yred,'ro',color='red')
plt.plot(3,5,'ro',color='green',markersize=15)
plt.axis([-0.5,10,-0.5,10])
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(x,y)
pred=classifier.predict([3,5])
print(pred)
plt.show()