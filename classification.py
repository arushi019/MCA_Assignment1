import os
import pickle
classes = {}
ct = 0
classified = {}
path = 'images/'
for file in os.listdir(path):
    s = file.split('_')
    name = s[0]
    if name in classes.keys():
        classified[file] = classes[name]
    else:
        classes[name] = ct
        ct += 1
        classified[file] = classes[name]
f = open('classified.pkl','wb')
pickle.dump(classified,f)
f.close()
    
