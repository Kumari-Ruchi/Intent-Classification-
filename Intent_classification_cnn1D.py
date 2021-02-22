#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential,Model, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding,Input
from keras.layers import Dropout,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,LeakyReLU
from keras.layers import SpatialDropout1D,Conv1D,MaxPooling1D,GRU,BatchNormalization
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


# In[6]:


def load_data(filename):
    df=pd.read_csv(filename,encoding='latin1',names=['Intent','Sentences'])
    print(df.head())
    intent=df['Intent']
    unique_intent=list(set(intent))
    sentences=list(df['Sentences'])
    
    return (intent,unique_intent,sentences)


# In[7]:


intent,unique_intent,sentences=load_data('/home/administrator/Downloads/intent_classification/atis_intents_train.csv')


# In[8]:


intent


# In[9]:


nltk.download('stopwords')
nltk.download('punkt')


# In[10]:


stemmer=LancasterStemmer()
stop_words = set(stopwords.words('english'))


# In[11]:


def cleaning(sentences):
    words=[]
    for s in sentences:
        clean = re.sub(r'[^a-zA-Z0-9]', " ",s)
        w=word_tokenize(clean)
        tokens_without_sw = [word for word in w if not word in stop_words]
        words.append([stemmer.stem(i.lower()) for i in tokens_without_sw])

    return words           


# In[12]:


cleaned_word=cleaning(sentences)
print(len(cleaned_word))
print(cleaned_word[:2])


# In[13]:


def creat_tokenizer(words,filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    tocken=Tokenizer(filters=filters)
    tocken.fit_on_texts(words)
    return tocken


# In[14]:


def max_len(words):
    return (len(max(words,key=len)))


# In[15]:


word_tockenizer=creat_tokenizer(cleaned_word)
vocab_size=len(word_tockenizer.word_index)+1
max_len=max_len(cleaned_word)

print("Vocab Size= %d and Max length= %d" % (vocab_size,max_len))


# In[16]:


def encoding_doc(tocken,words):
    return (tocken.texts_to_sequences(words))


# In[17]:


encoded_doc=encoding_doc(word_tockenizer,cleaned_word)


# In[28]:


encoded_doc[:5]


# In[18]:


def padding_doc(encoded_doc,max_len):
    return (pad_sequences(encoded_doc,maxlen=max_len,padding="post"))


# In[19]:


padded_doc=padding_doc(encoded_doc,max_len)


# In[29]:


padded_doc[:5]


# In[21]:


output_tokenizer = creat_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')


# In[22]:


output_tokenizer.word_index


# In[23]:


encoded_output = encoding_doc(output_tokenizer, intent)


# In[24]:


encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)


# In[25]:


encoded_output.shape


# In[26]:


def one_hot(encode):
    o = OneHotEncoder(sparse = False)
    return(o.fit_transform(encode))


# In[23]:


output_one_hot=one_hot(encoded_output)


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)


# In[26]:


print("shape of train_X = %s and train_Y = %s" % (train_X.shape,train_Y.shape))
print("shape of val_X=%s and val_Y=%s"%(val_X.shape,val_Y.shape))


# In[27]:


def create_model(vocab_size, max_length):
      model = Sequential()
      model.add(Embedding(vocab_size,128, input_length = max_length, trainable = False))
      model.add(SpatialDropout1D(0.5))
      model.add(Conv1D(filters=32, kernel_size=8,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      model.add(MaxPooling1D(pool_size=2))
      model.add(Bidirectional(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
      model.add(SpatialDropout1D(0.5))
      model.add(Conv1D(filters=32, kernel_size=8,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
      model.add(LeakyReLU(alpha=0.2))
      model.add(MaxPooling1D(pool_size=2))
      model.add(Bidirectional(LSTM(128,dropout=0.5, recurrent_dropout=0.5)))
      model.add(Dense(8,activation='softmax'))
      

      return model


# In[28]:


model = create_model(vocab_size, max_len)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()


# In[29]:


filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])


# In[34]:


model=load_model("model.h5")


# In[35]:


def predictions(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = word_tockenizer.texts_to_sequences(test_word)
  print(test_word)
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
 
  x = padding_doc(test_ls, max_len)
  
  pred = model.predict_proba(x)
  
  
  return pred


# In[36]:


def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
 
  for i in range(pred.shape[1]):
    print("%s has confidence = %s" % (classes[i], (predictions[i])))


# In[40]:


Text="what flights travel from las vegas to los angeles "
pred=predictions(Text)
get_final_output(pred,unique_intent)


# In[ ]:





# In[ ]:




