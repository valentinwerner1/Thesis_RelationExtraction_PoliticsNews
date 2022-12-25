FILE=C:/Users/svawe/Thesis_RelationExtraction_PoliticsNews/soft-data/src/preprocess/in_data
OUT=C:/Users/svawe/Thesis_RelationExtraction_PoliticsNews/soft-data/src/preprocess/out_data
STANFORD_PROPERTY=C:/Users/svawe/Thesis_RelationExtraction_PoliticsNews/soft-data/src/preprocess/StanfordCoreNLP-english.properties
udpipePath=C:/Users/svawe/udpipe-1.0.0-bin/udpipe-1.0.0-bin/bin-win64
languageModel=C:/Users/svawe/udpipe-1.0.0-bin/udpipe-1.0.0-bin/models/udpipe-ud-1.2-160523/english-ud-1.2-160523.udpipe
STANFORD_CORENLP=C:/Users/svawe/stanford-corenlp-4.5.0
FILENAME=Sample_english_doc.xml

echo "Call Stanford CoreNLP to do sentence splitting..."
java -cp "$STANFORD_CORENLP/*" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLP -props ${STANFORD_PROPERTY} -file ${FILE}/$FILENAME -outputFormat text -outputDirectory ${FILE}

echo "Generate sentence xml file..."
python preprocess_doc.py ${FILE}/$FILENAME
SFILENAME=$FILENAME-sent.xml.conll

echo "Call udpipe to do pos tagging and dependency parsing..."
${udpipePath}/udpipe --tag --parse --outfile=${FILE}/$SFILENAME.predpos.pred --input=conllu $languageModel ${FILE}/$SFILENAME

echo "Ouput parsed xml file..."
PFILENAME=$FILENAME-sent.xml
python preprocess_parse.py ${FILE}/$PFILENAME ${OUT}/$FILENAME.parsed.xml

echo "Removing intermediate files..."
rm ${FILE}/$FILENAME.out
rm ${FILE}/$PFILENAME
rm ${FILE}/$SFILENAME
rm ${FILE}/$SFILENAME.predpos.pred