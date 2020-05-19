
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_laplace
from skimage.transform import resize
from scipy.ndimage.filters import generic_filter
import matplotlib.pyplot as plt
from skimage import draw
import os
import pickle


# In[73]:
path = 'images/'
file_set = set(os.listdir())
d_final = {}
for name in os.listdir(path):
    #if name[:-4]+"_blob.jpg" in file_set:
    #    continue
    #print(name)
    file = 'images/'+name
    image = cv2.imread(file,0)


    # In[74]:


    image = image/255


    # In[76]:


    image


    # In[56]:


    sizes = list(range(2,11))


    # In[57]:


    g_images = []
    for i in sizes:
        temp = gaussian_laplace(image,i)
        g_images.append(temp)


    # In[59]:


    scale_space = np.zeros((9,image.shape[0],image.shape[1]),dtype = np.float64)
    for i in range(9):
        scale_space[i] = g_images[i]


    # In[60]:


    d = []
    for i in range(9):
        s = set()
        Rmax = generic_filter(g_images[i], np.max, footprint=np.ones((3, 3)))
        Rmax[Rmax != g_images[i]] = 0 
        v = Rmax[Rmax != 0]
        #print(v)
        x, y = np.nonzero(Rmax)
        x_ = []
        y_ = []
        for j in range(len(x)):
            if v[j]>0.001:
                temp = [x[j],y[j],v[j]]
                d.append(temp)


    # In[62]:


    d = sorted(d,key = lambda k:k[2],reverse = True)
    d_final[name] = d[:100]
    fff = open('log_pickle.pkl','wb')
    pickle.dump(d_final,fff)
    fff.close()
    print(name)

    # In[88]:


    """temp_image = image.copy()
    for i in d[:400]:
        xp = i[0]
        yp = i[1]
        #print(xp,yp)
        rr, cc = draw.circle(xp, yp, radius=10, shape=image.shape)
        temp_image[rr, cc] = 2
        #cv2.circle(image,(xp,yp),3,[0,255,0],2)
        #plt.plot(rr,cc,'o',color = 'red')
    #plt.show()
    plt.imshow(temp_image,cmap = 'gray')
    plt.axis('off')
    plt.savefig(name[:-4]+"_blob.jpg",bbox_inches = 'tight',dpi = 80)
    #print(i)"""


