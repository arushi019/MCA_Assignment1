from scipy.spatial.distance import cosine
import math
import os
import pickle
import numpy as np
import time
fff = open('classified.pkl','rb')
classified = pickle.load(fff)
fff.close()

fff = open("color_c.pkl","rb")
d = pickle.load(fff)
fff.close()

query_path = 'train/query/'
queries = {}
for file in os.listdir(query_path):
    fff = open(query_path+file,'r')
    for i in fff:
        j = i.split()[0][5:]
        queries[file.split('.')[0]] = j
#print(queries)

truth_path = 'train/ground_truth/'
good = {}
ok = {}
junk = {}
for file in os.listdir(truth_path):
    fff = open(truth_path+file,'r')
    s = set()
    for i in fff:
        s.add(i[:-1]+'.jpg')
    temp = file.split('.')[0]
    if "good" in file:
        good[temp[:-5]] = s
    if "ok" in file:
        ok[temp[:-3]] = s
    if "junk" in file:
        junk[temp[:-5]] = s
#print(good)
#print(ok)
#print(junk)
print("Loading done")
for query in queries.keys():
    start = time.time()
    q = queries[query]
    distance = {}
    for i in classified.keys():
      if i=='color_c.pkl':
        continue
      val = cosine(np.reshape(d[q+".jpg"],(-1)),np.reshape(d[i],(-1)))
      if classified[q+".jpg"] != classified[i]:
        val = math.sqrt(val)
      distance[i] = val
    #print(str(distance)[:100])
    sorted_distance = sorted(distance,key = lambda k: distance[k])
    #print(sorted_distance[:10])
    g = 0
    o = 0
    j = 0
    g_d = good[query[:-6]]
    o_d = ok[query[:-6]]
    j_d = junk[query[:-6]]
    for i in sorted_distance[:30]:
        #print(i)
        if i in g_d:
            g+=1
        if i in o_d:
            o+=1
        if i in j_d:
            j+=1
    end1 = time.time()
    #print(query,g,o,j)
    precision_g = g/30
    recall_g = g/len(g_d)
    precision_o = o/30
    recall_o = o/len(o_d)
    precision_j = j/30
    recall_j = j/len(j_d)
    #print(q)
    #print("Good",precision_g,recall_g)
    #print("Ok",precision_o,recall_o)
    #print("Junk",precision_j,recall_j)
    end2 = time.time()
    print(end1-start,end2-start,len(g_d),len(o_d),len(j_d))


