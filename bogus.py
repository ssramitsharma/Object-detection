
# coding: utf-8

# In[9]:


import os
import pandas as pd


# In[5]:


k = os.listdir()
k


# In[6]:



l = os.listdir("Images_png/")
cnt = len(l)
ms = os.path.abspath("Images_png/")
while cnt:
    
    k = os.listdir("Images_png/"+ l[cnt-1] )
    m = os.path.abspath("Images_png/"+ l[cnt - 1])+"/"
    print(m,"nose")
    print(k,'toes')
    for i in k:
        old_file = os.path.join(m, i)
        new_file = "/home/ramit/Downloads/bogus/"+ m.split("/")[6]+"_"+i
        os.rename(old_file, new_file)
        
        #print(old_file,"horse")
        

    cnt = cnt - 1


# In[8]:


os.listdir()


# In[12]:


df = pd.read_csv("DL_info.csv")
k = list(df["File_name"])
k


# In[14]:


os.path.abspath("DL_info.csv")


# In[15]:


m = '/home/ramit/Downloads/bogus'


# In[17]:


t = os.listdir(m)


# In[18]:


t


# In[19]:


for i in t:
    if i in k:
        print("Yes")

