import numpy as np
from math import log2


def kld(p, q):
	ans=0
	for i in p:
		for j in q:
			ans+=i*log2(i/j)
	return ans

if __name__ == '__main__':
	p=[.5, .25, .25]
	q=[.33, .33, .33]
	print(kld(p, q))
	print(kld(q, p))
