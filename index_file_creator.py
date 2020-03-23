import argparse
import os
from tqdm import tqdm

def create_indexed_files(opdir, train_file, val_file, test_file):
    import json
    import pickle
    from collections import defaultdict
    from nltk.corpus import wordnet as wn
    from nltk.tokenize import word_tokenize
    import numpy as np

    #Load custom candidate-wordnet file
    wn_dict = pickle.load(open(os.path.join(opdir, 'candidatesWN30.p'), 'rb'))
    pos_dict = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}


    i_token_to_i_id = defaultdict(int)
    i_token_to_i_id['UNK_TOKEN'] = 0
    count = 1 # 0th row is UNK_TOKEN
    
    train_file = os.path.join(opdir, train_file)
    val_file = os.path.join(opdir, val_file)
    test_file = [os.path.join(opdir, t) for t in test_file]
    data_files = [y for x in [[train_file], [val_file], test_file] for y in x]

    # First make the dictionary for i_token to
    for files in data_files:

        data = json.load(open(files))

        for sents in data:
            sent_toks = sents['original'].strip().split()
            for toks in sent_toks:
                if toks.strip() not in i_token_to_i_id:
                    i_token_to_i_id[toks.strip()] = count
                    count += 1

    with open(os.path.join(opdir, 'i_token_to_i_id.p'), 'wb') as handle:
        pickle.dump(i_token_to_i_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    i_id_to_i_token = dict([[v,k] for k,v in list(i_token_to_i_id.items())])

    with open(os.path.join(opdir, 'i_id_to_i_token.p'), 'wb') as handle:
        pickle.dump(i_id_to_i_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    o_token_to_o_id = defaultdict(int)

    for files in [train_file]:

        data = json.load(open(files))

        for sents in data:
            sent_toks = sents['annotated'].strip().split()
            for toks in sent_toks:
                if toks.strip() not in i_token_to_i_id:
                    if toks.strip() not in o_token_to_o_id:
                        o_token_to_o_id[toks.strip()] = count
                        count += 1

    with open(os.path.join(opdir, 'o_token_to_o_id.p'), 'wb') as handle:
        pickle.dump(o_token_to_o_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    o_id_to_o_token = dict([[v,k] for k,v in list(o_token_to_o_id.items())])

    with open(os.path.join(opdir, 'o_id_to_o_token.p'), 'wb') as handle:
        pickle.dump(o_id_to_o_token, handle, protocol=pickle.HIGHEST_PROTOCOL)


    o_token_to_o_id_remainingWordNet = {}


    i_id_to_candidate_wn_o_id = defaultdict(list)
    for files in data_files:

        data = json.load(open(files))

        for sents in data:
            x = sents['original'].strip().split()
            y = sents['annotated'].strip().split()
            for i in range(0,len(sents['offsets'])):
                offset_temp = sents['offsets'][i]
                stem_temp = sents['stems'][i]
                pos_temp = sents['pos'][i]

                if (stem_temp, pos_dict[pos_temp]) in wn_dict:
                    all_synsets = wn_dict[(stem_temp, pos_dict[pos_temp])]
                else:
                    all_synsets = []

                all_synsets_temp = []
                for ele in all_synsets:
                    if ele in o_token_to_o_id:
                        all_synsets_temp.append(o_token_to_o_id[ele])
                    else:
                        if ele not in o_token_to_o_id_remainingWordNet:
                            o_token_to_o_id_remainingWordNet[ele] = count
                            count+=1
                        all_synsets_temp.append(o_token_to_o_id_remainingWordNet[ele])

                #i_id_to_candidate_wn_o_id[i_token_to_i_id[x[offset_temp]]] = all_synsets_temp
                i_id_to_candidate_wn_o_id[(stem_temp, pos_dict[pos_temp])] = all_synsets_temp



    with open(os.path.join(opdir, 'o_token_to_o_id_remainingWordNet.p'), 'wb') as handle:
        pickle.dump(o_token_to_o_id_remainingWordNet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    o_id_remainingWordNet_to_o_token = dict([[v,k] for k,v in list(o_token_to_o_id_remainingWordNet.items())])

    with open(os.path.join(opdir, 'o_id_remainingWordNet_to_o_token.p'), 'wb') as handle:
        pickle.dump(o_id_remainingWordNet_to_o_token, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(opdir, 'i_id_to_candidate_wn_o_id.p'), 'wb') as handle:
        pickle.dump(i_id_to_candidate_wn_o_id, handle, protocol=pickle.HIGHEST_PROTOCOL)                


    i_id_to_candidate_train_o_id = defaultdict(list)
    for files in [train_file]:

        data = json.load(open(files))

        for sents in data:
            x = sents['original'].strip().split()
            y = sents['annotated'].strip().split()
            for i in range(0,len(sents['offsets'])):
                offset_temp = sents['offsets'][i]
                stem_temp = sents['stems'][i]
                pos_temp = sents['pos'][i]

                if (stem_temp, pos_dict[pos_temp]) in wn_dict:
                    all_synsets = wn_dict[(stem_temp, pos_dict[pos_temp])]
                else:
                    all_synsets = []
                # all_synsets = [a.name() for a in wn.synsets(x[i].replace('-','_'))] # bacause wordnet contains underscore separated words
                all_synsets_temp = []
                for ele in all_synsets:
                    if ele in o_token_to_o_id:
                        all_synsets_temp.append(o_token_to_o_id[ele])
                    else:
                        pass #all_synsets_temp.append(o_token_to_o_id_remainingWordNet[ele])

                #i_id_to_candidate_train_o_id[i_token_to_i_id[x[offset_temp]]] = all_synsets_temp
                i_id_to_candidate_train_o_id[(stem_temp, pos_dict[pos_temp])] = all_synsets_temp

    with open(os.path.join(opdir, 'i_id_to_candidate_train_o_id.p'), 'wb') as handle:
        pickle.dump(i_id_to_candidate_train_o_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for files in data_files:
        
        data = json.load(open(files))
        final_indexed_data = []
        for sents in data:
            indexed_orig_sent = ""
            for toks in sents['original'].strip().split():
                if toks in i_token_to_i_id:
                    indexed_orig_sent = indexed_orig_sent + " " + str(i_token_to_i_id[toks])
            indexed_anno_sent = ""
            for toks in sents['annotated'].strip().split():
                if toks in o_token_to_o_id:
                    indexed_anno_sent = indexed_anno_sent + " " + str(o_token_to_o_id[toks])
                elif toks in i_token_to_i_id:
                    indexed_anno_sent = indexed_anno_sent + " " + str(i_token_to_i_id[toks])
                elif toks in o_token_to_o_id_remainingWordNet:
                    indexed_anno_sent = indexed_anno_sent + " " + str(o_token_to_o_id_remainingWordNet[toks])
                else: # We need to think more for this
                    indexed_anno_sent = indexed_anno_sent + " " + str(0)

            final_indexed_data.append({
                'original' : indexed_orig_sent,
                'annotated' : indexed_anno_sent,
                'doc_offset' : sents['doc_offset'],
                'offsets' : sents['offsets'],
                'stems': sents['stems'],
                'pos': sents['pos']
                })

        fname = "_".join(files.split('_')[:-1])+"_indexed.json"
        with open(fname, 'w') as outfile:
            json.dump(final_indexed_data, outfile)

if(__name__=='__main__'):
    parser = argparse.ArgumentParser(description='This script takes in the train/val/test unindexed json file and gives out 8 dictionaries and corresponding indexed files')
    parser.add_argument('--train_file', type=str,
                        help='train file name')
    parser.add_argument('--val_file', type=str,
                        help='val file name')
    parser.add_argument('--test_file', type=str, action = 'append',
                        help='test file name')
    parser.add_argument('--opdir', type=str, default='./temp',
                        help='output dir path')
    args = parser.parse_args()

    create_indexed_files(args.opdir, args.train_file, args.val_file, args.test_file)
