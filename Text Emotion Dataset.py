#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
train_data_raw= pd.read_excel("text-emotion-training-dataset.xlsx")


# In[14]:


train_data_raw


# In[15]:


train_data = pd.DataFrame(train_data_raw["Text_Emotion"].str.split(';',1).tolist(), columns = ["text","emotion"])


# In[16]:


train_data


# In[17]:


train_data["emotion"].unique()


# In[18]:


encode_emotion = {"anger":0,"fear":1,"joy":2,"love":3,"sadness":4,"surprise":5}


# In[19]:


train_data.replace(encode_emotion,inplace= True)


# In[74]:


train_data


# In[21]:


training_sentence = []
training_label = []
for i in range(len(train_data)):
    sentence = train_data.loc[i,"text"]
    training_sentence.append(sentence)
    
    label= train_data.loc[i,"emotion"]
    training_label.append(label)


# In[24]:


#printing a random sentence and emotion
print(training_sentence[71],training_label[71])


# In[26]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 


# In[49]:


vocab_size = 10000
embedding_dim = 16
oov_tok = "<OOV>"
training_size = 20000

tokenizer = Tokenizer(num_words = vocab_size,oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentence)
word_index = tokenizer.word_index
word_index["the"]


# In[30]:


training_sequence = tokenizer.texts_to_sequences(training_sentence)
print(training_sequence[1])
print(training_sequence[4])


# In[38]:


#padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

padding_type = 'post'
max_length = 100
trunc_type = 'post'
training_padded = pad_sequences(training_sequence , maxlen = max_length,padding = padding_type,truncating = trunc_type)
training_padded[0]


# In[39]:


import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_label)


# In[40]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM , Dense
from tensorflow.keras.layers import Conv1D , Dropout,MaxPooling1D


# In[41]:


model = tf.keras.Sequential([
    Embedding(vocab_size,embedding_dim,input_length=max_length),
    Dropout(0.2),
    Conv1D(filters = 256, kernel_size = 3, activation = "relu"),
    MaxPooling1D(pool_size = 3),
    
    Conv1D(filters = 128, kernel_size = 3, activation = "relu"),
    MaxPooling1D(pool_size = 3),
    
    LSTM(128),
    
    Dense(128, activation = 'relu'),
    Dropout(0.2),
    Dense(64, activation = 'relu'),
    Dense(6,activation = 'softmax')])

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics =['accuracy'])


# In[42]:


model.summary()


# In[43]:


num_epochs = 30
history = model.fit(training_padded,training_labels,epochs = num_epochs,verbose = 2)


# In[44]:


model.save("Text_Emotion.h5")


# In[60]:


def find_key_by_value(my_dict, value_to_find):
    for key, value in my_dict.items():
        if value == value_to_find:
            return key
    return None 


# In[122]:


import speech_recognition as sr

def take_speech_input():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)

    try:
        sentence = recognizer.recognize_google(audio)
        return sentence
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Error with the request; {e}")
        return None


# In[119]:


speech = [speech_input]
print(type(speech))


# In[135]:


speech_input = take_speech_input()
if speech_input:
    # Now you can use the 'speech_input' variable in your program.
    print("Your speech input is:", speech_input)
else:
    print("Speech input not captured.")
speech = [speech_input]
sequences = tokenizer.texts_to_sequences(speech)
padded = pad_sequences(sequences,maxlen = max_length,padding = padding_type,truncating = trunc_type)

result = model.predict(padded)
predict_class = np.argmax(result,axis =1)
for i in predict_class:
    found_key = find_key_by_value(encode_emotion,i)
    print("Emotion is:",found_key)
#{"anger":0,"fear":1,"joy":2,"love":3,"sadness":4,"surprise":5}

