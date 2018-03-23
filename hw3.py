from nltk.stem.porter import *
#from operator import itemgetter, attrgetter
import math
import re
import collections

inputdoc=[]
inputlist=[]
wordlist=[]
#output=[]
stemmer = PorterStemmer()
dictfeature=[]
dictionary=dict()
AlldocinClass=[]
dictforclass=dict()

#前處理的function
def prefun(filename):
	#呼叫txt檔，並進行tokenization與小寫處理
	f = open(filename,'r')
	strlist = f.read()
	f.close()
	wordlist=[]

	strlist = re.sub(r'\d+',"",strlist)
	strlist = strlist.lower()
	strlist = re.sub('[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+',"",strlist)
	tokens = strlist.split()
	#tokens = strlist.replace(".","").replace(",","").replace("'","").replace("?","").replace("!","").replace('"',"").replace("`","").lower().split()

	#讀取stopwords list並存成陣列
	f = open('stopwords.txt','r')
	stopWords = f.read()
	f.close()

	tmp=[]
	#使用nltk套件的Porter Algorithm進行stemming 並存入txt檔案
	for i in tokens:
		tmp.append(stemmer.stem(i))

	#進行stopwords 的處理
	for j in tmp:
		if j not in stopWords:
			wordlist.append(j)
	#		output.append(j)
	#for word in wordlist
	#	if word in dictionary2:
	#		dictionary2[word]=dictionary2.get(word)+1

	return wordlist

def makedictionary(list):
    for i in list:
        dictionary[i]=dictionary.get(i,0)+1

#main
f = open('training.txt','r')

#print(inputlist)
#讀入input文件
for line in f.readlines():
	inputlist.append(line.split())
#將training document 讀入並整理出所有的token
for i in range(len(inputlist)):
	tmplist=[]
	for j in range(1,16):
		num = inputlist[i][j]
		wordlist=prefun("./IRTM/"+str(num)+".txt")
		makedictionary(wordlist)
		inputdoc.append(num)

		for k in wordlist:
			tmplist.append(k)
	AlldocinClass.append(tmplist)

#print(AlldocinClass)
#取前500個顯著的feature
x=1
for key,values in sorted(dictionary.items(),key=lambda  item:item[1], reverse=True):
	if x == 501:
		break
	else:
		dictfeature.append(key)
		x=x+1
#print(dictfeature)

#training phase
v=[]
tct=collections.defaultdict(dict)
probt=collections.defaultdict(dict)
for i in range(len(inputlist)):
	for j in AlldocinClass[i]:
		if j in dictfeature:
			v.append(j)

for i in range(len(inputlist)):
	prior=15 / 1095
	sigma=0
	for t in v:
		tct[t][i]=AlldocinClass[i].count(t)

	for t in v:
		sigma += tct[t][i]+1

	for t in v:
		probt[t][i] = tct[t][i]+1 / sigma

#testing phase
#1095份
f = open('R05725032.txt','w')
f.write("doc_id	class_id\n")

score=collections.defaultdict(dict)
for n in range(1,1096):
	if str(n) not in inputdoc:
		wordlist=prefun("./IRTM/"+str(n)+".txt")
		for i in range(len(inputlist)):
			score[n][i]= math.log10(prior)
			for t in wordlist:
				for key,values in sorted(probt.items(),key=lambda item:item[0]):
					if key == t:
						score[n][i] += math.log10(values[i])
		
		ans=max(score[n].items(), key=lambda x:x[1])[0]+1
		f.write(str(n)+"	"+str(ans))
		f.write("\n")
f.close()


				
		

		
