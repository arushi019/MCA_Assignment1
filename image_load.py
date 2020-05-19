import cv2
import numpy as np
import os
import pickle

path = 'images/'
images = {}
for file in os.listdir(path):
    img = cv2.imread(path+file,0)
    img = cv2.resize(img,(256,256))
    images[file] = img
    print(file)
f = open('load_images_256.pkl','wb')
pickle.dump(images,f)
f.close()
print("pickle done")
