import pickle
import numpy as np
import cv2
import os

file = open('load_images_256.pkl','rb')
images = pickle.load(file)
file.close()
print("images loaded")
surf = cv2.xfeatures2d.SURF_create()
surf_features = {}
for i in images.keys():
    kp,des = surf.detectAndCompute(images[i],None)
    surf_features[i] = des
    print(i)
file = open('surf_descriptors.pkl','wb')
pickle.dump(surf_features,file)
file.close()
print("pickle done")
