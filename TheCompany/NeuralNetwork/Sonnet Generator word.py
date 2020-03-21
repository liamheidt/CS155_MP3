import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
File = open("./data/AllSonnets.txt","r") 
seq='long'
#seq='short'
DDICT={'long':{'seqLength':10,'path':'Word Checkpoints Long/','Pattern':'my world my dying days behind for then i see'},'short':{'seqLength':4,'path':'Word Checkpoints/','Pattern':'my world my dying'}}
DICT=DDICT[seq]
DataLines = File.readlines()
File.close()
seqLength=DICT['seqLength']

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
model.add(Lambda(lambda x: x / .001))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


#%% 
#filename='Checkpoints - Copy/'+os.listdir('Checkpoints - Copy')[-1]
filename=DICT['path']+os.listdir(DICT['path'])[-1]
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
num2word=dict((num,word) for num, word in enumerate(Vocab))

start = np.random.randint(0, len(Xdata)-1)
#pattern = Xdata[start]
pattern=list(np.random.randint(0,3203,DICT['seqLength']))
#Pattern='my world my dying days behind for then i see'
#pattern=[word2num[i] for i in DICT['Pattern'].split(' ')]
print("Seed:",end='')
print("\"", ''.join([num2word[value]+' ' for value in pattern]), "\"")
output=''
counter=0
result=0
# generate characters
while counter<14:
#for i in range(200):
  x = np.reshape(pattern, (1, len(pattern), 1))/normalize
  prediction = model.predict(x, verbose=0)
  index=np.random.choice(len(prediction[0]),p=prediction[0])
#  index = prediction.argmax()
  oldresult=result
  result = num2word[index]
  if oldresult=='\n' and result =='\n':
    continue
  if result=='\n':
    counter+=1
  else:
    result+=' '
  seq_in = [num2word[value] for value in pattern]
  output+=result
  #  print(result,end='')
  pattern.append(index)
  pattern = pattern[1:len(pattern)]
  
print(output)    
  