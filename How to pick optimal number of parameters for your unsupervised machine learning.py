# -*- coding: utf-8 -*-
"""
Created on Sun May  9 06:03:38 2021

@author: abc
"""

import numpy as np
import cv2

#read our image
img = cv2.imread("alloy.jpg")

#reshape our image
img2 = img.reshape((-1,3))

#define gaussian mixture model
from sklearn.mixture import GaussianMixture as GMM

#define number of range of components
n_components = np.arange(1,10)

#Create model and fit it
gmm_model = [GMM(n, covariance_type='tied').fit(img2) for n in n_components]


#plot our model using BIC/AIC 
from matplotlib import pyplot as plt
plt.plot(n_components, [m.bic(img2) for m in gmm_model], label='BIC')
plt.xlabel('n_components')


                             #THANK YOU
