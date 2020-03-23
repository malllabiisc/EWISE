import os
from tqdm import tqdm
import numpy as np
import pickle
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import pdb

import torch

class Preprocessor():
    def __init__(self, wn18_path=""):
        if wn18_path:
            cached_path = os.path.join(wn18_path, 'cached_wn18_definitions.p')
        else:
            cached_path = 'cached_processed_definitions.p'

        if os.path.exists(cached_path):
            print ("Loading from ", cached_path)
            self.__dict__ = pickle.load(open(cached_path, 'rb'))
            return

        if wn18_path:
            self.procecss_wn18_definitions(wn18_path)
        else:
            self.process_definitions()

        self.tokenized_definitions = [self.tokenize(sentence) for sentence in self.definitions]
        self.word_vec = self.build_vocab(self.tokenized_definitions)

        with open(cached_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def procecss_wn18_definitions(self, wn18_path=""):
        definitions_file = os.path.join(wn18_path, "wordnet-mlj12-definitions.txt")
        with open(definitions_file) as f:
            lines = f.readlines()

        pos_conversion_map = {'NN':'n', 'VB':'v', 'JJ':'a', 'RB':'r'}
        self.definition_map = {}
        n_empty_definitions = 0
        for line in tqdm(lines):
            synset_offset, synset_tag, _ = line.split('\t')
            #definition_map[synset_idx.strip()] = definition.strip()
            #fetch definition from wordnet
            pos = synset_tag.split('_')[-2]
            wn_ss = wn.synset_from_pos_and_offset(pos_conversion_map[pos], int(synset_offset))
            definition = wn_ss.definition().strip()
            if len(definition) == 0:
                n_empty_definitions = n_empty_definitions + 1

            self.definition_map[synset_offset.strip()] = definition

        print ("#Empty definitions {}/{}".format(n_empty_definitions, len(self.definition_map)))
        synsets = sorted(self.definition_map.keys())
        self.synset_to_idx = {v:i for i,v in enumerate(synsets)}
        self.idx_to_synset = {v:i for i,v in self.synset_to_idx.items()}
        self.definitions = [self.definition_map[k] for k in synsets]

    def process_definitions(self):
        self.definition_map = {}
        self.lemmakey_to_synset = {}
        n_empty_definitions = 0
        print ("Processing definitions")
        all_synsets = wn.all_synsets()
        for s in tqdm(all_synsets):
            definition = s.definition().strip()
            if len(definition) == 0:
                n_empty_definitions = n_empty_definitions + 1

            self.definition_map[s.name()] = definition

            lemmas = s.lemmas()
            for lemma in lemmas:
                key = lemma.key()
                self.lemmakey_to_synset[key] = s.name()

        print ("#Empty definitions {}/{}".format(n_empty_definitions, len(self.definition_map)))

        synsets = sorted(self.definition_map.keys())
        #self.synset_to_idx = {v:i for i,v in enumerate(self.synset_to_definition.keys())}
        self.synset_to_idx = {v:i for i,v in enumerate(synsets)}
        self.idx_to_synset = {v:i for i,v in self.synset_to_idx.items()}
        self.definitions = [self.definition_map[k] for k in synsets]
    
    def get_batch(self, synsets, emb_dim=300):
        indices = [self.synset_to_idx[synset] for synset in synsets]
        sentences = [self.tokenized_definitions[idx] for idx in indices]
        sentences = [[w for w in sentence if w in self.word_vec] for sentence in sentences]

        lengths = np.array([len(x) for x in sentences])
        max_len = np.max(lengths)
        embed = np.zeros((max_len, len(sentences), emb_dim))

        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                embed[j, i, :] = self.word_vec[sentences[i][j]]

        return torch.from_numpy(embed).float(), lengths

    def tokenize(self, sentence):
        return ['<s>'] + word_tokenize(sentence) + ['</s>']

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<p>'] = ''
        return word_dict

    def get_glove(self, word_dict, glove_path):
        # create word_vec with glove vectors
        glove = pickle.load(open(glove_path, 'rb'))
        word_vec = {}
        for word in word_dict:
            if word in glove:
                word_vec[word] = glove[word] 
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))
        return word_vec

    def build_vocab(self, sentences, glove_path='../external/glove.p'):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_glove(word_dict, glove_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

if __name__ == '__main__':
    P = Preprocessor('../external/wordnet-mlj12')
    synsets = ['03964744', '00260881', '02199712']
    embeddings, lengths = P.get_batch(synsets)
    print (embeddings.size())

    P = Preprocessor()
    synsets = ['able.a.01', 'unable.a.01', 'abaxial.a.01']
    embeddings, lengths = P.get_batch(synsets)
    print (embeddings.size())
