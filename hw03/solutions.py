import numpy as np
from itertools import product
import re
import csv
import pandas as pd
import pdb


def hw03():
	#2a
	d='the sun rises in the east and sets in the west'
	words=d.strip().split(' ')
	N=len(words)
	c={}
	for w in words:
		if w not in c:
			c[w]=1
		else:
			c[w]+=1
	print(f'p\tword')
	for w in c.keys():
		print(f'{c[w]/N:.2f}\t{w}')
	print()

	#2b
	V=[
		'a', 
		'the',
		'from',
		'retrieval',
		'sun',
		'rises',
		'in',
		'BM25',
		'east',
		'sets',
		'and',
		'west'
	]
	c={w:0 for w in V}
	for w in words:
		c[w]+=1

	print(f'{"*"*10}')
	print(f'p\tword')
	for w in c.keys():
		print(f'{c[w]/N:.2f}\t{w}')
	print()

	#3a
	c={}
	for i in range(len(words)):
		if i==0:
			key = f'{words[i]} | #'
		else:
			key = f'{words[i]} | {words[i-1]}'

		if key not in c:
			c[key]=1
		else:
			c[key]+=1
	denom=sum(c.values())
	for key in c.keys():
		print(f'p({key})\t{c[key]/denom:.2f}')
	

	#5
	fname='state-of-the-union.txt'
	with open(fname, 'r') as f:
		corpus=f.read()


	#corpus=corpus.strip().replace('\n',' ').lower()
	#corpus=re.sub('[,.!?\"\';:-]', '', corpus)
	corpus=corpus.lower()
	reader=csv.reader(corpus)
	corpus=[doc for doc in reader]

	corpus=pd.read_csv(
		fname, 
		header=0,
		names=['year','speech']
	)
	
	#ignore the year column and make all documents lower case
	corpus=[doc.lower().replace('\n', ' ') for doc in corpus['speech']]
	corpus=[re.sub(r'[\'\",.?!:;\(\)\[\]\{\}]', '', doc) for doc in corpus]

	print(corpus[1])



if __name__ == '__main__':
	hw03()

