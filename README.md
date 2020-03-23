# EWISE

Reference code for [ACL2019](http://acl2019.org/) paper [Zero-shot Word Sense Disambiguation using Sense Definition Embeddings](https://www.aclweb.org/anthology/P19-1568/).
*EWISE*[1] (Extended WSD Incorporating Sense Embeddings) is a principled framework to learn from a combination of sense-annotated data, dictionary definitions and lexical knowledge bases.

<p align="center">
  <img align="center" src="https://github.com/malllabiisc/EWISE/blob/master/images/architecture.png" alt="...">
</p>

We have used the [WSD evalauation framework](http://lcl.uniroma1.it/wsdeval)[2] for training and evaluation.

## Dependencies
The code was written with, or depends on:
* Python 3.6
* Pytorch 1.4.0
* NLTK 3.4.5
* [WSD evalauation framework](http://lcl.uniroma1.it/wsdeval)[2]

## Running the code
1. Create a virtualenv and install dependecies
      ```bash
      virtualenv -p python3.6 env
      source env/bin/activate
      pip install -r requirements.txt
      python -m nltk.downloader wordnet
      python -m spacy download en
      ```         
1. Fetch data and pre-process. This will create pre-processed files in data folder.
      ```bash
      bash fetch_data.sh  
      bash preprocess.sh data
      ```     
1.  * To train ConvE embeddings, change directory to the ```conve``` folder and refer to the [README](./conve/README.md) in that folder. Generate embeddings for the WSD task:
      ```bash
      python generate_output_embeddings.py ./conve/saved_embeddings/embeddings.npz data conve_embeddings  
      ```    
    * Alternatively, to use pre-trained embeddings, copy the pre-trained conve embeddings (```o_id_embedding_conve_embeddings.npz```) to the ```data``` folder.
1.  Train a WSD model. This saves the model with best dev set score at ```./saved_models/model.pt```.
      ```bash
      CUDA_VISIBLE_DEVICES=0 python wsd_main.py --cuda --dropout 0.5 --epochs 200 --input_directory ./data --scorer ./ --output_embedding customnpz-o_id_embedding_conve_embeddings.npz --train semcor --val semeval2007 --lr 0.0001 --predict_on_unseen --save ./saved_models/model.pt
      ```
1. Test a WSD model (the model is assumed to saved at ```./saved_models/model.pt```.
      ```bash
      CUDA_VISIBLE_DEVICES=0 python wsd_main.py --cuda --dropout 0.5 --epochs 0 --input_directory ./data --scorer ./ --output_embedding customnpz-o_id_embedding_conve_embeddings.npz --train semcor --val semeval2007 --lr 0.0001 --predict_on_unseen --evaluate --pretrained ./saved_models/model.pt
      ```
      
## Pre-trained embeddings and models
All files are shared at https://drive.google.com/drive/folders/1NSrOx4ZY9Zx957RANFO90RX9daqIDElR
Uncompress model files using gunzip before using.
A & B would suffice if only training/evaluating a WSD model.

A. Pre-trained conve embeddings: ```o_id_embedding_conve_embeddings.npz```

B. Pre-trained model: ```model.pt.gz``` (F1 score on ALL dataset: 72.1)

C. Pre-trained ConvE model: ```WN18RR_conve_0.2_0.3__defn.model.gz```


An earlier version contained some code for weighted cross entropy loss (now enabled only by the ```--weighted_loss``` flag). The scheme wasn't really helpful and is not recommended. However, a pre-trained model for the same is shared: ```model_weighted.pt.gz``` (F1 score on ALL dataset: 72.1)
      
## Citation
If you use this code, please consider [citing](https://www.aclweb.org/anthology/P19-1568.bib):

[1] Kumar, Sawan, Sharmistha Jat, Karan Saxena, and Partha Talukdar. "Zero-shot word sense disambiguation using sense definition embeddings." In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 5670-5681. 2019.

## References
[2] Alessandro Raganato, Jose Camacho-Collados, and Roberto Navigli. 2017. Word sense disambiguation: A unified evaluation framework and empirical comparison. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 99–110, Valencia, Spain. Association for Computational Linguistics.

## Contact
For any clarification, comments, or suggestions please create an issue or contact sawankumar@iisc.ac.in
