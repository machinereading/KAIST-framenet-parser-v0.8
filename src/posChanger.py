
# coding: utf-8

# In[2]:

def posChanger(pos): 
    if pos.startswith('N'):
        pos = 'n'
    elif pos == 'VV':
        pos = 'v'
    elif pos == 'VA':
        pos = 'a'
    else:
        pass
    return pos

