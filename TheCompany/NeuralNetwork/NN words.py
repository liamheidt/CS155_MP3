import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
File = open("./data/AllSonnets.txt","r") 
DataLines = File.readlines()
File.close()


#remove punctuation
punctuation=['.', ',', ';', ':', '?', '!','(',')']
for i,line in enumerate(DataLines):
 line=line.translate({ord(i):None for i in punctuation})
 line=line.replace('\n',' \n ') 
 DataLines[i]=line 
Data=''.join(DataLines)
DataList=Data.split(' ')
Vocab=np.unique(DataList)
word2num=dict((word, num) for num, word in enumerate(Vocab))
#%%
seqLength=4
Xdata=[]
Ydata=[]
for i in range(0,len(DataList)-seqLength):
    x=DataList[i:i+seqLength]
    y=DataList[i+seqLength]
    Xdata.append([word2num[word] for word in x])
    Ydata.append(word2num[y])
    
#%%
X=np.reshape(Xdata,np.array(Xdata).shape+(1,))
normalize=float(X.max())
X=X/normalize
Y=np_utils.to_categorical(Ydata)
#%%
model=Sequential()
model.add(LSTM(150,input_shape=X.shape[1:]))
#model.add(LSTM(200,input_shape=X.shape[1:],return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(200))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
filename='Word Checkpoints/'+os.listdir('Word Checkpoints')[-1]
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


filepath="Word Checkpoints/long-word-weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


model.fit(X, Y, epochs=20000, batch_size=32, callbacks=callbacks_list,verbose=1)


