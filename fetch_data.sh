mkdir external
cd external

wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip

#WSD evaluation framework
wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
unzip WSD_Evaluation_Framework
cp ./WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java ../

#wordnet training data
wget -O wordnet-mlj12.tar.gz https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz
tar xvzf wordnet-mlj12.tar.gz
cd ..

javac Scorer.java

#create glove dict
python create_glove_dict.py ./external/glove.840B.300d.txt ./external/glove.p

mkdir saved_models
