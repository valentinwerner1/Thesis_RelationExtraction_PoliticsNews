import xml.etree.ElementTree as ET
import sys

# xml_in = input("Please specify relative path (from current directory) of -sent-xml:  ")
xml_in = sys.argv[1]
conll_parsed = xml_in+".conll.predpos.pred"
#xml_out = xml_in.replace(".xml","_parsed.xml")
xml_out = sys.argv[2]

#get all sentences parsed by UDPipe
conll = open(conll_parsed,'r',encoding='utf-8')
lines = conll.readlines()
conll.close()

#reunite sentences that belong together
sent_reformed = []
sent = ""
for word in lines:
    if word != "\n": sent += word
    else: 
        sent_reformed.append(sent)
        sent = ""

tree = ET.parse(xml_in)     #parse the xml file created by preprocess_doc
root = tree.getroot()       #get the roots of xml tree, can be iterated

#add a subelement "parse" with parsed sentence from UDPipe
for idx, xmlement in enumerate(root):
    ET.SubElement(xmlement,"Parse").text = sent_reformed[idx]   
tree.write(xml_out,encoding='utf-8',xml_declaration=True) 