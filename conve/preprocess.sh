#!/bin/bash
mkdir data
mkdir data/WN18RR
mkdir saved_models
mkdir saved_embeddings
tar -xvf WN18RR.tar.gz -C data/WN18RR
python wrangle_KG.py WN18RR
