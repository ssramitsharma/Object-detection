
# coding: utf-8

# In[1]:


import os
from zipfile import ZipFile


# In[2]:


k = os.listdir()
k


# In[3]:


os.path.abspath('Images_png_55.zip')


# In[4]:


def extract(a):
    print('Extract all files in ZIP to current directory')
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(a, 'r') as zipObj:
       # Extract all the contents of zip file in current directory
        zipObj.extractall()
        zipObj.close()
        


# In[5]:


for i in k:
    if i.endswith(".zip"):
        extract(os.path.abspath(i))

