# EWISE

Reference code for [ACL2019](http://acl2019.org/) paper "Zero-shot Word Sense Disambiguation using Sense Definition Embeddings".
*EWISE*[1] (Extended WSD Incorporating Sense Embeddings) is a principled framework to learn from a combination of sense-annotated data, dictionary definitions and lexical knowledge bases.

<p align="center">
  <img align="center" src="https://github.com/malllabiisc/EWISE/blob/master/images/architecture.png" alt="...">
</p>

We have used the [WSD evalauation framework](http://lcl.uniroma1.it/wsdeval)[2] for training and evaluation.

## Dependencies
The code was written with, or depends on:
* Python 2
* Pytorch 0.4.0
* NLTK 3.2.5
* [WSD evalauation framework](http://lcl.uniroma1.it/wsdeval)[2]

## Running the code
As of now, the code requires a working directory with the following files:
* ```i_id_to_i_token``` : dictionary (pickle file) -- input token index to input token
* ```i_id_embedding``` : numpy array (pickle file) -- input embedding matrix  
* ```o_id_to_o_token``` : dictionary (pickle file) -- output token index to output token (for train tokens; output tokens indices include and begin with input token indices) 
* ```o_id_remainingWordNet_to_o_token``` : dictionary (pickle file) -- output token index to output token (for output tokens outsode the training set, obtained from WordNet)
* ```i_id_to_candidate_wn_o_id``` : dictionary (pickle file) -- input token index to a list of candidate output token indices (obtained from WordNet)
* ```i_id_to_candidate_train_o_id``` : dictionary (pickle file) -- input token index to a list of candidate output token indices (obtained from training set)
* ```{semcor, semeval2007, senseval2, senseval3, semeval2013, semeval2015, ALL}_indexed.json``` : list (pickle file) -- list of indexed train/test examples (a dictionary for each sentence) created from the [WSD evalauation framework](http://lcl.uniroma1.it/wsdeval)[2] with the following fields:
  * ```original``` : input token indices
  * ```annotated``` : output token indices
  * ```offsets``` : offsets in ```original```/```annotated``` for annotated tokens
  * ```pos``` : POS tags for annotated words (equal in length to ```offsets```)
  * ```stems``` : Stems for annotated words (equal in length to ```offsets```) 
  * ```doc_offset``` : Tag used for making predictions for words in the sentence

Additionally, depending on the encoding method used, it requires:
* $output_embeddings : numpy array (npz) -- output embedding matrix 

Finally, the Scorer script from the [WSD evalauation framework](http://lcl.uniroma1.it/wsdeval)[2] needs to be present on the machine. The directory path to the Socrer needs to be provided.

To test an existing model:
```bash
python wsd_main.py --cuda --input_directory $working_directory --scorer $path_to_scorer --output_embedding customnpz-$output_embeddings --train semcor --val semeval2007 --predict_on_unseen --evaluate --pretrained $model_path
```
To train a model:
```bash
python wsd_main.py --cuda --dropout 0.5 --epochs $num_epochs --input_directory $working_directory    --scorer ./ --output_embedding customnpz-$output_embeddings  --train semcor --val semeval2007 --lr 0.0001 --predict_on_unseen  --save $save_path
```

## Citation
If you use this code, please consider citing:

[1] Sawan Kumar, Sharmistha Jat, Karan Saxena and Partha Talukdar. 2019. Zero-shot Word Sense Disambiguation using Sense Definition Embeddings. Accepted to the 57th Annual Meeting of the Association for Computational Linguistics (ACL) (proceedings in press).

## References
[2] Alessandro Raganato, Jose Camacho-Collados, and Roberto Navigli. 2017. Word sense disambiguation: A unified evaluation framework and empirical comparison. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 99–110, Valencia, Spain. Association for Computational Linguistics.

## Contact
For any clarification, comments, or suggestions please create an issue or contact sawankumar@iisc.ac.in
