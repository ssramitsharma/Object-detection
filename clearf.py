
# coding: utf-8

# In[1]:


import shutil
import os


# In[3]:



for i in os.listdir():
    if 'pneu' in i:
        shutil.move('/home/rsharm2s/'+i,'/home/rsharm2s/bogus')
