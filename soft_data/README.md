## Credits where Credits is due

This code is heavily oriented on the idea of https://github.com/eventdata/UDPetrarch-for-Python3

Some lines of code have been copied directly, many have been rewritten for the purpose of clearity and simplicity

To run this you need to install CoreNLP and UDPipe and specify their respective paths

## Preprocesing

For preprocessing go into /preprocess

The .sh file executes the whole pipeline, simply run

'''
sh preprocess_articles_from_xml.sh
'''

after specifying the paths inside .sh file (paths to all necessary directories).

Input format is XML, see /example files -> sample_english_doc.xml = input format; [...]-sent_parsed.xml = output format

## Annotation

The annotation process is build on PETRARCH. All dictionaries are not created by me, but by the researchers behind CAMEO and PETRARCH.

For more information see https://github.com/eventdata/UDPetrarch-for-Python3 and https://github.com/openeventdata/petrarch2 

