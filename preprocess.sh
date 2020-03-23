if [ -z $1 ]
then
    echo "please specify a path to store the data"
    exit
fi

DATAPATH=$1
WSDEVALFW=./external/WSD_Evaluation_Framework
testnames="semeval2007 senseval2 senseval3 semeval2013 semeval2015 ALL"
trainname="semcor"

mkdir $DATAPATH

#unindexed files
#test files
for testname in $testnames
do
    goldfile="$WSDEVALFW"/Evaluation_Datasets/"$testname"/"$testname".gold.key.txt
    xmlfile="$WSDEVALFW"/Evaluation_Datasets/"$testname"/"$testname".data.xml
    python readxml.py --goldfile "$goldfile" --xmlfile "$xmlfile" --opname "$testname"_unindexed --opdir "$DATAPATH"
    cp $goldfile $DATAPATH
done
#train data
goldfile="$WSDEVALFW"/Training_Corpora/SemCor/semcor.gold.key.txt
xmlfile="$WSDEVALFW"/Training_Corpora/SemCor/semcor.data.xml
python readxml.py --goldfile "$goldfile" --xmlfile "$xmlfile" --opname "$trainname"_unindexed --opdir "$DATAPATH"

#create candidate dictionary
python create_candidate_dictionary.py "$WSDEVALFW"/Data_Validation/candidatesWN30.txt "$DATAPATH"/candidatesWN30.p 

#indexed files
python index_file_creator.py --train_file semcor_unindexed.json --val_file semeval2007_unindexed.json --test_file senseval2_unindexed.json --test_file senseval3_unindexed.json --test_file semeval2013_unindexed.json --test_file semeval2015_unindexed.json --test_file ALL_unindexed.json --opdir "$DATAPATH"

python initialize_word_embeddings.py ./external/glove.p  "$DATAPATH"
