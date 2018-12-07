
# coding: utf-8

# In[11]:


import json
import posChanger
import sys
sys.path.insert(0,'../')
from koreanframenet import kfn
import re
from copy import deepcopy


# In[10]:


def target_identification_ko(conll):
    result = []
    frame = 'None'
    for i in conll:
        #print(i[0], i[2], i[3])
        lu1, lu2 = [],[]
        #print(i)
        lus =[]
        lex = i[2].split('+')[0].split('/')[0]
        pos = i[2].split('+')[0].split('/')[1]
        pos = posChanger.posChanger(pos)
        lemma = lex+'.'+pos
        lu1 = kfn.lus_by_lemma(lemma)
        #print(lu1)
        
        surfaceform = i[1]
        spc = [',','.','!','?']
        if len(surfaceform) > 1:
            if surfaceform[-1] in spc:
                surfaceform = re.sub('[,.?!]', '', surfaceform)
        lu2 = kfn.lus_by_surfaceform(surfaceform)
        lus = lu1+lu2
        lus = list(set(lus))
        
        pos = i[4].split('+')[0]
        pos = posChanger.posChanger(pos)
        lu_candis = []
        if len(lus) > 0:
            for lc in lus:
                lu_pos = lc.split('.')[1]
                #print('pos', pos, lu_pos)
                if pos == lu_pos:
                    lu_candi_list = lc.split('.')[:-1]
                    lu_candi = '.'.join(lu_candi_list)
                    lu_candis.append(lu_candi)
                    #print(lu_candi)
                    #if surfaceform[0] == lu_candi[0]:
                        #lu_candis.append(lu_candi)
        lu_candis = list(set(lu_candis))
#         print(lu_candis)
        if len(lu_candis) > 0:
            lu = False
            max = 0
            for j in lu_candis:
                lexu_list = kfn.lus_by_lu(j)
                for k in lexu_list:
                    count = len(k['ko_annotation_id'])
                    if count > max:
                        lu = j
                        max = count                
            lu_dict = {}
            lu_dict['token_id'] = i[0]
            lu_dict['lu'] = lu
            lu_with_frame = []
            for j in lus:
                lexu = j.split('.')[0] + '.' + j.split('.')[1]
                if lexu == lu:
                    lu_with_frame.append(j)
            lu_dict['lu_with_frame'] = lu_with_frame
            if lu != False:
                result.append(lu_dict)
    return result


# In[12]:


def target_identification_en(conll, enlu):
    result = []
    new_conll = deepcopy(conll)
    
    targets = []
    for token_ix in range(len(new_conll)):
        token = new_conll[token_ix]
        pos = token[5]
        if pos.startswith('N'):
            pos = 'n'
        elif pos.startswith('V'):
            pos = 'v'
        elif pos.startswith('RB'):
            pos = 'a'        
            
        lem = token[3]+'.'+pos
        
        for lu in enlu:
            if lem == lu:
                tu = (token_ix, lu)
                targets.append(tu)
                break
    for i in targets:
        target_ix, lu = i[0],i[1]
        
        new_conll = deepcopy(conll)
        
        for token_ix in range(len(new_conll)):
            
            token = new_conll[token_ix]
            if token_ix == target_ix:
                token.append(lu)
            else:
                token.append('_')
        result.append(new_conll)
    return result
            

