#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from itertools import repeat
import numpy as np 
import csv


# In[21]:


from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import spacy
nlp = spacy.load("en_core_web_sm")
import re
import os
import seaborn as sns
import keras.layers as layers
from keras.models import Model
from keras import backend as K
np.random.seed(10)


# In[22]:


model = hub.load("USEmodel")
#module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
#model = hub.load(module_url)
#print ("module %s loaded" % module_url)
def embed(input):
    return model(input)


# In[23]:


def run_sts_benchmark(batch):
    #with tf.Session() as session:
    #    session.run([tf.global_variables_initializer(), tf.tables_initializer()])        
    sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(batch['sent_1'].tolist())), axis=1)
    sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(batch['sent_2'].tolist())), axis=1)
    #print (sts_encode1)
    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    #print ("Cbaaaa---------------------------------------")
    #print (cosine_similarities)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities)
   
    """Returns the similarity scores"""
    return scores


# In[31]:


def simscore(policylist, lawseg,policy,law):
    lawlist=list(law[law['Category'] == lawseg].Segment)

    policyFinal=[]
    for p in policylist:
        policyFinal.extend(repeat(p, len(lawlist)))
    lawFinal=[]
    lawnum=[]
    numl=np.arange(1,len(lawlist)+1,1)
    #print (numl)
    #numl=numl.tolist()
    for j in range(len(policylist)):
        #for l in lawlist
        lawFinal.extend(lawlist)
        lawnum.extend(numl)
    #print (lawnum)
    #lawFinal.append(repeat(lawlist,len(policylist)))
    #print((policyFinal[0]))
    #print((lawFinal[0]))
    polnum=policyFinal
    policyFinal = pd.Series(policyFinal)
    lawFinal = pd.Series(lawFinal)
    correlate = {'sent_1': policyFinal,'sent_2': lawFinal}
    #print(correlate)
    #df = pd.DataFrame(correlate)
    #print (df)
    score=run_sts_benchmark(correlate)
    #print (score)
    return score,lawnum,polnum


# In[38]:


def compScore(scoret,maxs,mins):
    #with sess.as_default():
    
    #print (tf.size(score))
    #comp=np.array([0]*len(score))
    #for i in range(len(score)):
    #    if score[i]<0.25:
    #        comp[i]=100
    score=scoret.numpy()
    score=score*(-1)
    #if pdpc thresholds
    if (maxs<mins):
        score[score > mins]=mins
        score[score<maxs]=maxs
        score=(mins-score)/(mins-maxs)  
    else:   #if gdpr  
        score[score < mins]=mins
        score[score>maxs]=maxs
        score=(maxs-score)/(maxs-mins)
    #print (score.shape)
    score=score*100
    return score 


# In[41]:


def getCompliance(policyfile, lawfile, lawname):
    maxs=0.0
    mins=0.0
    if (lawname=="GDPR"):
        maxs=0.6
        mins=0.25
    else:
        maxs=0.09
        mins=0.5
    avgscore=[0.0]*4
    policy=pd.read_csv(policyfile,engine ='python')
    #print (policy)
    law=pd.read_csv(lawfile)
    #print (law)
    Policycat=['firstparty','thirdparty','userchoice','dataretention','useraccess']
    lawseg=[1,3,4,5]
    #For gdpt1polic PDPA purpose-1 #first party and third party
    PolicyList1=(list(policy[policy[Policycat[0]] == 1].Segment))
    PolicyList1.extend(list(policy[policy[Policycat[1]] == 1].Segment))
    pl1=np.unique(PolicyList1)
    if len(pl1)>0:
        score1,lawnum,polnum=simscore(pl1,lawseg[0],policy,law)
        #print (score1)
        comp1=compScore(score1,maxs,mins)
        avgscore[0]=sum(comp1)/len(comp1)
        with open("gdpr1.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PolicySegment","SubSegment","Score"])
            for i in range(len(comp1)):
                writer.writerow([polnum[i], lawnum[i],comp1[i]])
            file.close()
    else:
        comp1=[]
        avgscore[0]=0.0
        with open("gdpr1.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PolicySegment","SubSegment","Score"])
            file.close()
    
    #gdpr3 - PDPA consent - User choice
    PolicyList1=(list(policy[policy[Policycat[2]] == 1].Segment))
    if len(PolicyList1)>0:
        score1,lawnum,polnum=simscore(PolicyList1,lawseg[1],policy,law)
        comp1=compScore(score1,maxs,mins)
        avgscore[1]=sum(comp1)/len(comp1)
        with open("gdpr3.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PolicySegment","SubSegment","Score"])
            for i in range(len(comp1)):
                writer.writerow([polnum[i], lawnum[i],comp1[i]])
            file.close()
    else:
        comp1=[]
        avgscore[1]=0.0
        with open("gdpr3.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PolicySegment","SubSegment","Score"])
            file.close()
        
    #gdpr4 - PDPA correction - User access
    PolicyList1=(list(policy[policy[Policycat[4]] == 1].Segment))
    if len(PolicyList1)>0:
        score1,lawnum,polnum=simscore(PolicyList1,lawseg[3],policy,law)
        comp1=compScore(score1,maxs,mins)
        avgscore[2]=sum(comp1)/len(comp1)
        with open("gdpr4.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PolicySegment","SubSegment","Score"])
            for i in range(len(comp1)):
                writer.writerow([polnum[i], lawnum[i],comp1[i]])
            file.close()
    else:
        comp1=[]
        avgscore[2]=0.0
        with open("gdpr4.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PolicySegment","SubSegment","Score"])
            file.close()

    #gdpr5 PDPA retention
    PolicyList1=(list(policy[policy[Policycat[3]] == 1].Segment))
    if len(PolicyList1)>0:
        score1,lawnum,polnum=simscore(PolicyList1,lawseg[2],policy,law)
        comp1=compScore(score1,maxs,mins)
        avgscore[3]=sum(comp1)/len(comp1)
        with open("gdpr5.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PolicySegment","SubSegment","Score"])
            for i in range(len(comp1)):
                writer.writerow([polnum[i], lawnum[i],comp1[i]])
            file.close()
    else:
        comp1=[]
        avgscore[3]=0.0
        with open("gdpr5.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["PolicySegment","SubSegment","Score"])
            file.close()
    #for p in pl1:
    #    print (p)
    #    print("-----------")
    totalscore=sum(avgscore)/len(avgscore)
    
    return {
        "compliance": round(totalscore,2),
        "GDPR1": round(avgscore[0],2), #PDPA PURPOSE
        "GDPR3": round(avgscore[2],2), #PDPA PDPA correction 
        "GDPR4" :round(avgscore[3],2), #PDPA Data Retention
        "GDPR2": round(avgscore[1],2), #PDPA CONSENT
       }
    


# In[42]:


#compliance("policycat.csv","GDPRSegments.csv")


# In[ ]: