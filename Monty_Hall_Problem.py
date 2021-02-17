#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from time import time as TT
from random import choice as ch
import numpy as np
ac = []
tc = []
N = []
st = TT()
for M in range(1,10): 
    st1 = TT()
    score = []
    runs = 0
    doors = [1,2,3]
    for K in range(1,M): 
        aset = []
        host = doors.copy()
        hbk = ch(host) 
        aset.append(hbk)
        print("The host knows the answer",hbk)
        player = doors.copy()
        px = ch(player) 
        aset.append(px)
        print ("Players first choice",px)
        chance = 0
        for i in host: 
            if i not in aset:
                chance = i
        print ("The elimination",chance)
        print (player)
        player.pop(player.index(chance))
        player.pop(player.index(px))
        print ("final answer",player)
        if player[0] == hbk:
            score.append(1)
        else:
            score.append(0)
        runs = K
        print ("\n\n")
    ac.append(np.mean(score))
    N.append(M)
    en1 = TT()
    tc.append(en1-st1)
en = TT()    
print ("Total time for Loop  ", en - st )


# In[ ]:




