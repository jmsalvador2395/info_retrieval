import numpy as np
from itertools import product


def upper_tri_indices(n):
	mask=np.full((n,n), True)
	mask=np.triu(mask)
	np.fill_diagonal(mask, False)
	x,y=np.indices((n,n))

	return x[mask], y[mask]
"""
generate indices for unique pairs
"""
def gen_indices(lw_indices):
	n=len(lw_indices)
	x,y=upper_tri_indices(n)

	#mask out irrelevant indices
	return lw_indices[x], lw_indices[y]
	#return lw_indices[x[mask]], lw_indices[y[mask]]

def get_topk(mat, k=10):
	
	#create index arrays
	x, y = np.indices(mat.shape)
	x = x.flatten()
	y = y.flatten()

	#get sorted args
	topk=np.argsort(mat.flatten())
	topk=np.flip(topk)

	#only keep first k indices
	x = x[topk[:k]]
	y = y[topk[:k]]

	return x, y

def print_largest_k(vocab, mat, metric, k=10):
	x, y=get_topk(mat, k)
	topbar=f'******************************** top {k} {metric}***********************************'
	print()
	print(topbar)
	if metric == 'counts':
		print('counts\t\tw1 idx\t\tw2 idx\t\tword pairs')
	else:
		print('mi\t\tw1 idx\t\tw2 idx\t\tword pairs')
	print('*'*len(topbar))

	for i in range(k):
		w1=vocab[x[i]]
		w2=vocab[y[i]]
		if metric == 'counts':
			print(f'{int(mat[x[i], y[i]])}\t\t{x[i]}\t\t{y[i]}\t\t{w1}, {w2}')
		else:
			print(f'{mat[x[i], y[i]]:.2f}\t\t{x[i]}\t\t{y[i]}\t\t{w1}, {w2}')

	print('*'*len(topbar))
	

if __name__ == '__main__':


	k=10

	#read in vocab
	fname='cacm.trec.filtered.txt'
	vocab=[]
	with open(fname, 'r') as f:
		line=f.readline()
		while(line):
			doc_words=line.strip().split(' ')
			for w in doc_words:
				if w not in vocab:
					vocab.append(w)
			line=f.readline()
	vocab=np.array(vocab)
	N=len(vocab)

	#initialize array for word pair counts
	pair_counts=np.zeros((N, N))
	word_counts=np.zeros(N)

	#count word pairs
	with open(fname, 'r') as f:
		line=f.readline()

		#loop for each document
		while(line):

			#split doc into individual words
			doc_words=np.array(line.strip().split(' '))

			#get unique indices of all words in the doc
			lw_indices=(vocab[:, None] == doc_words).argmax(axis=0)
			lw_indices=np.unique(lw_indices)

			#get indices
			if(len(lw_indices)>1):
				x,y = gen_indices(lw_indices)
				pair_counts[x, y]+=1
				pair_counts[y, x]+=1
				word_counts[lw_indices]+=1
		
			line=f.readline()

	
	pair_counts=np.triu(pair_counts)

	#topk_x, topk_y=get_topk(pair_counts, k)
	print_largest_k(vocab, pair_counts,'counts', k)


	topk_words=np.argsort(word_counts)[::-1]
	topk_words=topk_words[:k]

	"""
	compute mutual information
	"""

	mask=np.full((N,N), True)
	mask=np.triu(mask)
	np.fill_diagonal(mask, False)

	#compute probabilities
	p11=(pair_counts+.25)/(1+N)
	p1=(word_counts+.5)/(1+N)


	p0=1-p1
	p10=p1-p11
	p01=p1[:, None]-p11
	p00=p0-p01

	#compute mutual info
	x, y=upper_tri_indices(N)
	mi=np.zeros((N,N))

	#doing it manually because who tf cares
	mi+=p00*np.log2(p00/((p01+p00)*(p10+p00)))
	mi+=p01*np.log2(p01/((p00+p01)*(p11+p01)))
	mi+=p10*np.log2(p10/((p11+p10)*(p00+p10)))
	mi+=p11*np.log2(p11/((p10+p11)*(p01+p11)))
	mi=mi*mask

	#mi=mi[x, y]

	print_largest_k(vocab, mi,'mutual information', k)

	prog_mask=np.full((N,N), False)
	prog_idx=np.where(vocab == 'programming')[0][0]
	prog_mask[:, prog_idx]=True
	prog_mask[prog_idx, :]=True
	prog_mask=np.triu(prog_mask)
	np.fill_diagonal(prog_mask, False)

	print_largest_k(vocab, mi*prog_mask,'mutual information (programming)', k)

