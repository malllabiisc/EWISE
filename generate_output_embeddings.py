import sys
import os
import numpy as np
import pickle
from nltk.corpus import wordnet as wn

inpfile=sys.argv[1]
opdir=sys.argv[2]
opname=sys.argv[3]

d = np.load(inpfile)
embeddings = d['embeddings']
synsets = d['synsets']
print ('input', embeddings.shape)
emb_dim = embeddings.shape[1]
zeros = np.zeros(emb_dim)

synset_to_idx = {v:i for i,v in enumerate(synsets)}

o_id_to_o_token = pickle.load(open(os.path.join(opdir, 'o_id_to_o_token.p'), 'rb'))
i_id_to_i_token = pickle.load(open(os.path.join(opdir, 'i_id_to_i_token.p'), 'rb'))
i_id_embedding = pickle.load(open(os.path.join(opdir, 'i_id_embedding_glove.p'), 'rb'))
o_id_remainingWordNet_to_o_token = pickle.load(open(os.path.join(opdir, 'o_id_remainingWordNet_to_o_token.p'), 'rb'))

v_s_start = len(i_id_to_i_token)
v_s_length = len(o_id_to_o_token)
v_r_length = len(o_id_remainingWordNet_to_o_token)

output_embeddings = []
for i in range(0,v_s_start):
    output_embeddings.append(zeros)
for i in range(0,v_s_length):
    synset = wn.lemma_from_key(o_id_to_o_token[i+v_s_start]).synset().name()
    output_embeddings.append(embeddings[synset_to_idx[synset]])
for i in range(0,v_r_length):
    synset = wn.lemma_from_key(o_id_remainingWordNet_to_o_token[i+v_s_start+v_s_length]).synset().name()
    output_embeddings.append(embeddings[synset_to_idx[synset]])

output_embeddings = np.stack(output_embeddings, 0)
print ('output', output_embeddings.shape)
np.savez_compressed(os.path.join(opdir, 'o_id_embedding_{}.npz'.format(opname)), embeddings=output_embeddings)
