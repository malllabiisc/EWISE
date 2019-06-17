# coding: utf-8
from collections import Counter

pos_dict = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

def build_mfs(batches, use_stem=True, return_lemma_freq=False):
    all_words = []
    all_lemmas = []
    word_dict = {}
    word_freq = {}
    for batch_idx, batch in enumerate(batches):
        #print batch
        x,y,a,tag, stem, POStag = batch
        bsz = x.shape[0]
        seqlen = x.shape[1]
        x,y = x.T, y.T
        #print (x.shape, y.shape)
        for i in range(bsz):
            xi = x[:,i]
            yi = y[:,i]
            ai = a[i]
            stemi = stem[i]
            POStagi = POStag[i]
            #print (xi, yi, ai, stemi, POStagi)
            for pos_idx, pos in enumerate(ai):
                xidx = xi[pos]
                yidx = yi[pos]
                stemidx = stemi[pos_idx]
                POStagidx = POStagi[pos_idx]
                #use xidx, or (stemidx, POStagidx) to generate MFS
                if use_stem:
                    key = (stemidx, POStagidx)
                else:
                    key = xidx
                all_words.append(key)
                all_lemmas.append(yidx)
                if key not in word_dict:
                    word_dict[key] = [yidx]
                else:
                    word_dict[key].append(yidx)
    all_lemmas = Counter(all_lemmas)
    lcnt = sum(all_lemmas[i] for i in all_lemmas)
    for e in all_lemmas:
        all_lemmas[e] = all_lemmas[e] * 1.0/lcnt
    for word in word_dict:
        wC = Counter(word_dict[word])
        word_dict[word] = wC.most_common(1)[0][0]
        wcnt_sum = sum([wC[i] for i in wC])
        wC = {i:wC[i]*1.0/wcnt_sum for i in wC}
        word_freq[word] = wC
        #print (word_freq[word])
    if return_lemma_freq:
        return word_dict, word_freq, all_lemmas
    return word_dict, word_freq
