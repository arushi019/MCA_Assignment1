
# coding: utf-8

# In[4]:


import cv2
import matplotlib.pyplot as plt
import pickle
import os
# In[24]:

path = 'images/'
d = {}
for name in os.listdir(path):
    file = 'images/'+name
    image = cv2.imread(file,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))


    # In[11]:


    distances = [1,3,5,7]


    # In[29]:


    rep_colors = []
    for i in range(0,256,30):
        for j in range(0,256,30):
            for k in range(0,256,30):
                rep_colors.append([i,j,k])


    # In[30]:


    len(rep_colors)


    # In[35]:


    def get_bin(color):
        temp = [i-i%30 for i in color]
        return temp


    # In[32]:


    color_index = {}
    index_color = {}
    for i in range(len(rep_colors)):
        color_index[str(rep_colors[i])] = i
        index_color[i] = rep_colors[i]


    # In[33]:


    def get_ngh(x,y,k,m,n):
        ngh = []
        if y-k>=0:
            for i in range(max(0,x-k),min(x+k,n)):
                ngh.append([y-k,i])
        if y+k<m:
            for i in range(max(0,x-k),min(x+k,n)):
                ngh.append([y+k,i])
        if x-k>=0:
            for i in range(max(0,y-k+1),min(y+k-1,m)):
                ngh.append([i,x-k])
        if x+k<n:
            for i in range(max(0,y-k+1),min(y+k-1,m)):
                ngh.append([i,x+k])
        return ngh


    # In[36]:


    vector = []
    for k in distances:
        colors = [0]*729
        count = 0
        for i in range(64):
            for j in range(64):
                ngh = get_ngh(i,j,k,64,64)
                for a in ngh:
                    col = image[a[0]][a[1]]
                    col = get_bin(col)
                    idx = color_index[str(col)]
                    colors[idx] += 1
                    count += 1
        colors = [i/count for i in colors]
        vector.append(colors)
    d[name] = vector
    fff = open('color_c.pkl','wb')
    pickle.dump(d,fff)
    fff.close()
    print(name)

