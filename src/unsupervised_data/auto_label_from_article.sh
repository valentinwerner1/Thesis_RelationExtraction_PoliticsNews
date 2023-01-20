STANFORD_CORENLP=C:/Users/svawe/stanford-corenlp-4.5.0
FILENAME=C:/Users/svawe/Thesis_RelationExtraction_PoliticsNews/src/unsupervised_data/data/raw
FILE_OUT=C:/Users/svawe/Thesis_RelationExtraction_PoliticsNews/src/unsupervised_data/data/out_data
PICOPATH=C:/Users/svawe/Thesis_RelationExtraction_PoliticsNews/src/unsupervised_data/src/add_labels/dictionaries/PETR.Internal.Coding.Ontology.txt
VERBPATH=C:/Users/svawe/Thesis_RelationExtraction_PoliticsNews/src/unsupervised_data/src/add_labels/dictionaries/newdict.txt

#echo "Preparing environment..."
#python -m pip install -r requirements.txt

echo "Turning csv into xml..."
python src/preprocess/csv_to_xml.py $FILENAME/$1 $FILE_OUT/$1.xml

echo "Call Stanford CoreNLP to do sentence splitting..."
java -cp "$STANFORD_CORENLP/*" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,cleanxml,ssplit -file $FILE_OUT/$1.xml -outputFormat text -outputDirectory $FILE_OUT

echo "Finding triplets in the sentences..."
python src/add_labels/sent_to_triplet.py $FILE_OUT/$1.xml.out $PICOPATH $VERBPATH $FILE_OUT/$1.done.csv

echo "Removing intermediate files..."
rm $FILE_OUT/$1.xml.out
rm $FILE_OUT/$1.xml
