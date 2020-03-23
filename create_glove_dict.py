import numpy as np
import sys
from tqdm import trange
import pickle

inpfile = sys.argv[1]
outfile = sys.argv[2]

embedding_dim = 300

with open(inpfile) as f:
    lines = f.readlines()

lines = [line.split() for line in lines]
lines = [line for line in lines if len(line)==301] 

words = [line[0] for line in lines]
nwords = len(words)

d = {}
for idx in trange(nwords):
    d[words[idx].strip()] = np.array(lines[idx][1:],  dtype=np.float)

print ('saving embeddings to', outfile)
with open(outfile, 'wb') as fout:
    pickle.dump(d, fout)
