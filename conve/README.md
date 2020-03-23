## Training a ConvE based encoder for definitions
This folder contains a copy of the necessary files and modifications over https://github.com/TimDettmers/ConvE.

## Steps to train embeddings
1. Preprocess files
    ```bash
    bash preprocess.sh
    ```
1. Run ConvE to get initial embeddings for entities and relations. Model is saved at ```saved_models/WN18RR_conve_0.2_0.3_.model```. When running for the first time, add the ```--preprocess``` flag.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python conve_main.py --model conve --data WN18RR  --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2  --lr 0.001 
    ```
1. Train ConvE with a definition encoder. Model is saved at ```saved_models/WN18RR_conve_0.2_0.3__defn.model```
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_definition_encoder.py --model conve --data WN18RR  --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2  --lr 0.0001 --batch-size 128 --test-batch-size 128  --epochs 500  --initialize  saved_models/WN18RR_conve_0.2_0.3_.model
    ```
1. Generate representations for definitions using trained model. Embeddings are saved at ```saved_embeddings/embeddings.npz```. These embeddings can now be used to train a WSD model following the instructions in the root directory.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_definition_encoder.py --model conve --data WN18RR  --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2  --lr 0.0001 --batch-size 128 --test-batch-size 128  --epochs 0  --resume --represent saved_embeddings/embeddings.npz
    ```

## LICENSE
Most of the files in this directory are copied with or without modifications from https://github.com/TimDettmers/ConvE. The corresponding LICENSE applies (also copied at LICENSE in this directory). The modifications are described in the correspoding files.
An exception is ```definition_preprocessor.py``` which is an entirey original code written for this project and the LICENSE in the root directory applies.
