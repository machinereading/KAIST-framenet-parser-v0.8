
# coding: utf-8

# In[39]:


# coding: utf-8

# In[31]:

import json
import os
from koreanframenet import kfn
import preprocessor

kolus, annos, s_annos = kfn.load_kfn()

def load_tsv(lines):
    result = []
    sent = []
    sent_ids = []
    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('#'):
            if line[1] == 's':
                sent_id = line.split(':')[1]
                sent_ids.append(sent_id)                
            pass
        else:
            if line != '':
                token = line.split('\t')
                sent.append(token)
            else:
                result.append(sent)
                sent = []
    sent_num = len(list(set(sent_ids)))
    return result, sent_num

def data_stat(language, USE_EXAM=False):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    if language == 'ko':
        path = dir_path+'/data/kofn/'
        training_dir = path+'training.tsv'
        test_dir = path+'test.tsv'
        dev_dir = path+'dev.tsv'
        exemplar_dir = path+'examplar.tsv'
    elif language == 'en':
        path = dir_path+'/data/fn1.5/'
        training_dir = path+'fn1.5.fulltext.train.syntaxnet.conll'
        test_dir = path+'fn1.5.test.syntaxnet.conll'
        dev_dir = path+'fn1.5.dev.syntaxnet.conll'
        exemplar_dir = path+'fn1.5.exemplar.train.syntaxnet.conll'
    else:
        path = dir_path+'/data/kofn/'
        training_dir = path+'training.tsv'
        test_dir = path+'test.tsv'
        dev_dir = path+'dev.tsv'
        exemplar_dir = path+'examplar.tsv'
    
    with open(training_dir,'r') as f:
        d = f.readlines()
        training, n_training = load_tsv(d)
        if language == 'en':
            with open(path+'fn1.5.fulltext.train.syntaxnet.conll.sents','r') as f:
                d = f.readlines()
            n_training = len(d)
        #print(len(training_fe))
    with open(test_dir,'r') as f:
        d = f.readlines()
        test, n_test = load_tsv(d)
        if language == 'en':
            with open(path+'fn1.5.test.syntaxnet.conll.sents','r') as f:
                d = f.readlines()
            n_test = len(d)
        #print(len(test))
    with open(dev_dir,'r') as f:
        d = f.readlines()
        dev, n_dev = load_tsv(d)
        if language == 'en':
            with open(path+'fn1.5.dev.syntaxnet.conll.sents','r') as f:
                d = f.readlines()
            n_dev = len(d)
        #print(len(training))
    
    if USE_EXAM:
        with open(exemplar_dir,'r') as f:
            d = f.readlines()
            exemplar, n_exemplar = load_tsv(d)
            if language == 'en':
                with open(path+'fn1.5.exemplar.train.syntaxnet.conll.sents','r') as f:
                    d = f.readlines()
                n_exemplar = len(d)
    else:
        exemplar = []
        n_exemplar = 0
        
#     print('# training_data')
#     print(' - number of sentences:', n_training)
#     print(' - number of annotations:', len(training), '\n')
    
#     print('# test_data')
#     print(' - number of sentences:', n_test)
#     print(' - number of annotations:', len(test), '\n')
    
#     print('# dev_data')
#     print(' - number of sentences:', n_dev)
#     print(' - number of annotations:', len(dev), '\n')
    
    if USE_EXAM:
        pass
#         print('# exemplar data')
#         print(' - number of sentences:', n_exemplar)
#         print(' - number of annotations:', len(exemplar), '\n')
    else:
        pass

def load_data(language, USE_EXAM=False):
    dir_path = os.path.dirname(os.path.abspath(__file__))
#     dir_path = '.'
    if language == 'ko':
        path = dir_path+'/data/kofn/'
        training_dir = path+'training.tsv'
        test_dir = path+'test.tsv'
        dev_dir = path+'dev.tsv'
        exemplar_dir = path+'examplar.tsv'
    elif language == 'en':
        path = dir_path+'/data/fn1.5/'
        training_dir = path+'fn1.5.fulltext.train.syntaxnet.conll'
        test_dir = path+'fn1.5.test.syntaxnet.conll'
        dev_dir = path+'fn1.5.dev.syntaxnet.conll'
        exemplar_dir = path+'fn1.5.exemplar.train.syntaxnet.conll'
    else:
        path = dir_path+'/data/kofn/'
        training_dir = path+'training.tsv'
        test_dir = path+'test.tsv'
        dev_dir = path+'dev.tsv'
        exemplar_dir = path+'examplar.tsv'
    
    print('### loading data now...')
    with open(training_dir,'r') as f:
        d = f.readlines()
        training, n_training = load_tsv(d)
        #print(len(training_fe))
    with open(test_dir,'r') as f:
        d = f.readlines()
        test, n_test = load_tsv(d)
        #print(len(test))
    with open(dev_dir,'r') as f:
        d = f.readlines()
        dev, n_dev = load_tsv(d)
        #print(len(training))
    if USE_EXAM:
        with open(exemplar_dir,'r') as f:
            d = f.readlines()
            exemplar, n_exemplar = load_tsv(d)
    else:
        exemplar = []
        n_exemplar = 0
        
#     print('# training_data')
#     print(' - number of sentences:', n_training)
#     print(' - number of annotations:', len(training), '\n')
    
#     print('# test_data')
#     print(' - number of sentences:', n_test)
#     print(' - number of annotations:', len(test), '\n')
    
#     print('# dev_data')
#     print(' - number of sentences:', n_dev)
#     print(' - number of annotations:', len(dev), '\n')
    
#     print('# exemplar data (from sejong)')
#     print(' - number of sentences:', n_exemplar)
#     print(' - number of annotations:', len(exemplar), '\n')
    
    return training, test, dev, exemplar
              
#training, test, training_fe = load_data()   


# In[ ]:


# In[60]:


from collections import Counter
def dummy(language='en'):
    training, test, dev, exemplar = load_data(language)
    data_all = training+test+dev+exemplar
    lufrmap, lufr_count_map, frargmap = {},{},{}
    for sent_list in data_all:
        for token in sent_list:
            lu, frame, fe = token[12],token[13],token[14]
            if lu != '_':
                if lu not in lufr_count_map:
                    frames = []
                    frames.append(frame)
                    lufr_count_map[lu] = frames
                else:
                    frames = lufr_count_map[lu]
                    frames.append(frame)
                    lufr_count_map[lu] = frames
    for i in lufr_count_map:
        count = dict(Counter(lufr_count_map[i]).items())
        lufr_count_map[i] = count
    
# dummy()


# In[66]:


# if data is changed, this is required
def gen_map_data(language):
    training, test, dev, exemplar = load_data(language)
    data_all = training+test+dev+exemplar
    lufrmap, lufr_count_map, frargmap = {},{},{}
    for sent_list in data_all:
        for token in sent_list:
            lu, frame, fe = token[12],token[13],token[14]
            if lu != '_':
                if lu not in lufrmap:
                    frames = []
                    frames.append(frame)
                    lufrmap[lu] = frames
                else:
                    frames = lufrmap[lu]
                    frames.append(frame)
                    frames = list(set(frames))
                    lufrmap[lu] = frames
                    
    for sent_list in data_all:
        for token in sent_list:
            lu, frame, fe = token[12],token[13],token[14]
            if lu != '_':
                if lu not in lufr_count_map:
                    frames = []
                    frames.append(frame)
                    lufr_count_map[lu] = frames
                else:
                    frames = lufr_count_map[lu]
                    frames.append(frame)
                    lufr_count_map[lu] = frames
    for i in lufr_count_map:
        count = dict(Counter(lufr_count_map[i]).items())
        lufr_count_map[i] = count                  
      
    for sent_list in data_all:
        args = []
        frame = False
        for token in sent_list:
            lu, f, fe = token[12],token[13],token[14]
            if f != '_':
                frame = f
            if fe != 'O':
                fe = fe.split('-')[1]
                args.append(fe)
                    
        args = list(set(args))
        if frame:
            if frame not in frargmap:
                fes = args
                frargmap[frame] = fes
            else:
                fes = frargmap[frame]
                fes = fes + args
                fes = list(set(fes))
                frargmap[frame] = fes
    print('### NUM OF LUS:',len(lufrmap))
    print('### NUM OF FRAMES:',len(frargmap))
    with open('./data/'+language+'.lufrmap.json','w') as f:
        json.dump(lufrmap, f, ensure_ascii=False, indent=4)
    with open('./data/'+language+'.lufr_count_map.json','w') as f:
        json.dump(lufr_count_map, f, ensure_ascii=False, indent=4)    
    with open('./data/'+language+'.frargmap.json','w') as f:
        json.dump(frargmap, f, ensure_ascii=False, indent=4)    
# gen_map_data('ko')


# In[65]:


def gen_map_data_with_count(language):
    training, test, dev, exemplar = load_data(language)
    data_all = training+test+dev+exemplar
    lufrmap, frargmap = {},{}
    for sent_list in data_all:
        for token in sent_list:
            lu, frame, fe = token[12],token[13],token[14]
            if lu != '_':
                if lu not in lufrmap:
                    frames = []
                    frames.append(frame)
                    lufrmap[lu] = frames
                else:
                    frames = lufrmap[lu]
                    frames.append(frame)
#                     frames = list(set(frames))
                    lufrmap[lu] = frames
    for sent_list in data_all:
        args = []
        frame = False
        for token in sent_list:
            lu, f, fe = token[12],token[13],token[14]
            if f != '_':
                frame = f
            if fe != 'O':
                if language == 'en':
                    fe = fe.split('-')[1]
                    args.append(fe)
        args = list(set(args))
        if frame:
            if frame not in frargmap:
                fes = args
                frargmap[frame] = fes
            else:
                fes = frargmap[frame]
                fes = fes + args
                fes = list(set(fes))
                frargmap[frame] = fes
    print('### NUM OF LUS:',len(lufrmap))
    print('### NUM OF FRAMES:',len(frargmap))
    with open('./data/'+language+'.lufrmap.json','w') as f:
        json.dump(lufrmap, f, ensure_ascii=False, indent=4)
    with open('./data/'+language+'.frargmap.json','w') as f:
        json.dump(frargmap, f, ensure_ascii=False, indent=4)

# gen_map_data_with_count('ko')


# In[46]:


# if data is changed, this is required
def gen_wv_voca(language):
    if language == 'en':
        with open('./data/glove.6B.100d.framevocab.txt','r') as f:
            d = f.readlines()
        word_to_wv = {}
        for i in d:
            i = i.strip()
            word = i.split(' ')[0]
            wv_list = i.split(' ')[1:]
            wv = ' '.join(wv_list)
            word_to_wv[word] = wv
    dir_path = './data/'+language+'.token2wv.json'
    with open(dir_path,'w') as f:
        json.dump(word_to_wv, f, ensure_ascii=False, indent=4)
    print('### NUM OF VOCA:', len(word_to_wv))


# In[20]:


def read_map(language):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = dir_path+'/data/'+language+'.'
    with open(path+'lufrmap.json','r') as f:
        lufrmap = json.load(f)
    with open(path+'frargmap.json','r') as f:
        frargmap = json.load(f)
    return lufrmap, frargmap
# lufrmap, frargmap = read_map()


# In[35]:


def eval_data(language):
    training, test, dev, exemplar = load_data(language)
        
    lus_in_training_data = []
    for sent_list in training:
        for token in sent_list:
            lu = False
            target, frame = token[12], token[13]
            if target != '_':
                lu = target+'.'+frame
                lus_in_training_data.append(lu)
    lus_in_training_data = list(set(lus_in_training_data))
    count, total = 0,0
    for sent_list in test:
        lu = False
        for token in sent_list:            
            target, frame = token[12], token[13]
            if target != '_':
                lu = target+'.'+frame
        if lu:
            if lu in lus_in_training_data:
                count += 1
            total += 1
    
    print('# LU coverage')
    print(total, 'lus are in TEST DATA')
    print(count, '('+str(round((count/total)*100, 2))+'%)''lus are covered by TRAINING DATA')

