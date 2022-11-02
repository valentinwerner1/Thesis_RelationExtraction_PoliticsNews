import xml.etree.ElementTree as ET
import sys

#inputxml = input("Please specify relative path (from current directory) of raw xml input:")
inputxml = sys.argv[1]
inputparsed = inputxml+".out"
outputfile = inputxml+"-sent.xml"
conll_out = outputfile+".conll"

docdict = {}
doctexts = []
output = []

tree = ET.iterparse(inputxml)

#the first for loop is written by UDPetrarch
i = 0
for event, elem in tree:
	if event == "end" and elem.tag == "Article":
		story = elem

		# Check to make sure all the proper XML attributes are included
		attribute_check = [key in story.attrib for key in ['date', 'id', 'mongoId','sentence', 'source']]
		if not attribute_check:
			print('Need to properly format your XML...')
			break

		entry_id = story.attrib['id'];
		date = story.attrib['date']
		date = date[0:date.find("T")].replace("-","") ## kill later
		source = story.attrib['source']

		text = story.find('Text').text
		if text is None:
			text = ""
		else:
			text = text.replace('\n', ' ').strip()

		if entry_id in docdict:
			print('id must be unique, this article is in document dictionary :'+entry_id)
			break

		docdict[i] = {'id':entry_id,'date':date,'source':source,'text':text}
		i += 1
		doctexts.append(text)
	
		elem.clear()

#parse all lines from CoreNLP sentence split
parsed = open(inputparsed)
parsedfile = parsed.readlines()
parsedlines = []

#Only keep those lines which have Sentence #n in the line before
for idx, text in enumerate(parsedfile):
    if text.startswith("Sentence #"):
        parsedlines.append(parsedfile[idx+1].replace('\n','').strip()) #clean text and append

out_dict = {}
#iterables for dict entries, sentence counts and count of sentences per article
dict = 0 
count = 0 
count_article = 0 
for line in parsedlines:
    if line in docdict[dict]["text"]:	#check if text is in dictionairy entry, if yes append to out_dict with meta data
        out_dict[count] = {"sentence_id":str(docdict[dict]["id"])+f"_{count_article}",
                           "date": docdict[dict]["date"],
                           "source": docdict[dict]["source"],
                           "sent": line}
        count_article += 1
        count += 1
    else:								#if not in dictionairy entry, it must be in the next, reset iterables and append to next key
        dict += 1
        count_article = 0
        if line in docdict[dict]["text"]:
            out_dict[count] = {"sentence_id":str(docdict[dict]["id"])+f"_{count_article}",
                               "date": docdict[dict]["date"],
                               "source": docdict[dict]["source"],
                               "sent": line}
            count_article += 1
            count += 1
        else: 
            print("couldn't find line")

#create new xml tree structur being <sentences><sentence id=... <text> text <\text> <\sentence> <sentences>
root = ET.Element("Sentences")			
for idx, a in enumerate(parsedlines):
	sentence = ET.SubElement(root,"Sentence", {
							"date":out_dict[idx]['date'],
							"source":out_dict[idx]['source'],
							"id":out_dict[idx]['sentence_id']
							})
	ET.SubElement(sentence,"Text").text = out_dict[idx]["sent"]
tree = ET.ElementTree(root)
tree.write(outputfile,'UTF-8',xml_declaration=True)

#needed for the entity parsing
file = open(conll_out, 'w') 
for idx, line in enumerate(parsedlines):
	if idx != 0: 
		file.write("\n")
	wc = 1 #initate word count at 1 
	for word in line.split(" "):
		file.write(f"{wc}\t{word}\t_\t_\t_\t_\t_\t_\t_\t_\n")
		wc += 1
file.close()
	