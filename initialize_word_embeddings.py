import sys
import os
import pickle
import numpy as np
import json

embeddingfile = sys.argv[1]
outputdir = sys.argv[2]

model = pickle.load(open(embeddingfile,'rb'))
print(("vectors loaded", len(model)))

i_id_to_i_token = pickle.load(open(os.path.join(outputdir, "i_id_to_i_token.p"),"rb"))

glove_list = [np.zeros(300)]

for i in range(0, len(i_id_to_i_token)):
    value = i_id_to_i_token[i]
    if value.strip() == "UNK_TOKEN":
        pass
    elif value.strip() in model:
        glove_list.append(model[value.strip()])
    elif '_' not in value:
        glove_list.append(np.zeros(300))
    else:
        if value.replace("_","-") in model:
            glove_list.append(model[value.replace("_","-")])
            continue
        toks = value.strip().split('-')
        temp_vec = np.zeros(300)
        for tok in toks:
            if tok.strip() in model:
                temp_vec = np.add(temp_vec, model[tok.strip()])
            else:
                pass
        glove_list.append(temp_vec)


i_id_embedding = np.array(glove_list)

print(i_id_embedding.shape)
with open(os.path.join(outputdir, 'i_id_embedding_glove.p'), 'wb') as handle:
    pickle.dump(i_id_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

