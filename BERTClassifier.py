#!/usr/bin/env python
# coding: utf-8

# In[50]:


#https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# USed tutorial given above
#hiii


# In[51]:


mod = ["model/firstparty","model/thirdparty","model/userchoice","model/useraccess","model/dataretention"]
#fileName= "Chevron.txt"
#filePath = ""+fileName


# In[58]:


import torch
from keras.preprocessing.sequence import pad_sequences
import re
import csv
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import os


# In[53]:


def Segmenter(filepath):    
    policySegments = []
    temp=""
    with open(filepath,encoding='utf-8') as policy:
        for line in policy:
        #print(line)
            if(line !='\n'):
                count = len(re.findall(r'\w+', line))
                #print (count)
                if(count < 20):
                    temp+=line
                else:
                    policySegments.append(line)
                    temp=""
    return policySegments


# In[54]:


def Predict(sentences,output_dir):
    #Creating the different models
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)
    
    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True,)
        input_ids.append(encoded_sent)
    MAX_LEN = 300
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0.0,truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [int(i>0) for i in seq]
        attention_masks.append(seq_mask) 
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    # Put model in evaluation mode
    model.eval()
    # Tracking variables 
    predictions = []
    # Predict 
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs=model(prediction_inputs.long(),token_type_ids=None,attention_mask=prediction_masks)
    logits = outputs[0]
    logits=logits.numpy()
    predictions.append(logits)
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    return flat_predictions


# In[55]:


def Categorize(strpol):
    #Segmenting Policies
    sentences=Segmenter(strpol)
    pred=[0] * 5
    for i in range(5):
        pred[i]=Predict(sentences,mod[i])
    #print (pred)
    #Writing results in file
    with open('policycat.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Segment","firstparty","thirdparty","userchoice","useraccess","dataretention"])
        for i in range(len(sentences)):
            writer.writerow([sentences[i], pred[0][i],pred[1][i],pred[2][i],pred[3][i],pred[4][i]])
        file.close()
    return 'policycat.csv'


# In[56]:


#pred=[0] * 5
#print (pred)


# In[57]:


#Categorize(filePath)


# In[ ]:




