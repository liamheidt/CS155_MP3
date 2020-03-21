import numpy as np
import pickle

DictData=np.loadtxt('data/Syllable_dictionary.txt',dtype='str',delimiter=',')
SyllDict={}
LessThanEq={}
for i in range(11):
  LessThanEq[i]={}
for word in DictData:
  SpaceInd=word.find(' ')
  Word=word[:SpaceInd]
  Syllables=word[SpaceInd+1:]
  if len(Syllables)==1:
    SyllDict[Word]={'normal':[int(Syllables)],'end':[int(Syllables)],'min':int(Syllables)}
  else:
    EndBehavior=Syllables.find('E')!=-1
    SpaceInd=Syllables.find(' ')
    S1=Syllables[:SpaceInd]
    S2=Syllables[SpaceInd+1:]
    if EndBehavior:
      is1=S1.find('E')==0
      if is1:
        SyllDict[Word]={'normal':[int(S2)],'end':[int(S1[-1])],'min': min([int(S2),int(S1[-1])])}
      else:
        SyllDict[Word]={'normal':[int(S1)],'end':[int(S2[-1])],'min':min([int(S1),int(S2[-1])])}
    else:
      SyllDict[Word]={'normal':[int(S1),int(S2)],'end':[int(S1),int(S2)],'min':min([int(S1),int(S2)])}  
  for i in np.arange(SyllDict[Word]['min'],11):
    LessThanEq[i][Word]=SyllDict[Word]

#File=open('SyllDict.pkl','wb')
#pickle.dump(SyllDict,File)
#File.close()
#File=open('LessThanEq.pkl','wb')
#pickle.dump(LessThanEq,File)
#File.close()
#

Pronunciation={}
NoStress={}

for line in open("cmudict-0.7b").readlines():
  line=line.strip()
  if line.startswith(';'): continue
  word, phones =line.split('  ')
  word=word.rstrip("(0123)").lower()
  if word not in Pronunciation:
    Pronunciation[word]=[]
    NoStress[word]=[]    
  Pronunciation[word].append(phones)
  NoStress[word].append(phones.translate({i+48:None for i in range(3)}))

for line in open("6393.dict").readlines():
  line=line.strip()
  word, phones =line.split('\t')
  word=word.rstrip("(0123)").lower()
  if word not in NoStress:
    NoStress[word]=[]    
  NoStress[word].append(phones)

#File=open('NoStress.pkl','wb')
#pickle.dump(NoStress,File)
#File.close()
#File=open('Pronounciation.pkl','wb')
#pickle.dump(Pronunciation,File)
#File.close()

#%%
#noPronounce=[]
#for word in SyllDict:
#  try:
#    print(NoStress[word])
#  except:
#    noPronounce.append(word)
##np.savetxt('NoPronounciation2',noPronounce,fmt='%s')
#        
    
#%% 
vowels=np.array([['AA','AE','AH','AO','AW','AY','EH','ER','EY','IH','IY','OW','OY','UH','UW']]).T
rhymes={}
for i,word in enumerate(SyllDict):
  if np.mod(i,30)==0:
    print('%i/%i'%(i,len(SyllDict)))
  rhymes[word]=[]
  for pronounce in NoStress[word]:
    Sylls=pronounce.split(' ')
    SyllArray=np.array([Sylls])
    vowelInd=(SyllArray==vowels).any(0).nonzero()[0][-1]-len(Sylls)
    RhymeSylls=Sylls[vowelInd:]

    for key, values in NoStress.items():
      rhyme=False
      for value in values:
        if len(value)>=-vowelInd:
          if value.split(' ')[vowelInd:]==RhymeSylls:
            rhyme=True
      if rhyme:
        if key not in rhymes[word]:
          rhymes[word].append(key)
  rhymes[word].remove(word)
File=open('Rhyme2.pkl','wb')
pickle.dump(rhymes,File)
File.close()
      
      
  
  


  
  
    
    
  
#SyllList=['None']*len(SyllDict)


