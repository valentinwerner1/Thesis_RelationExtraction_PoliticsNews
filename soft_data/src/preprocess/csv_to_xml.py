import sys
import pandas as pd
import xml.etree.ElementTree as ET

df = pd.read_csv(sys.argv[1])

outputfile = sys.argv[2]
root = ET.Element("Articles")			
for row in df.iterrows():
	art_xml = ET.SubElement(root,"Article", {
							#"date":row[1]["date"],
							"url":row[1]["link"],
							"paper":row[1]["paper"],
							})
							#"id":article[article_id]
	ET.SubElement(art_xml,"Text").text = str(row[1]["text"]).replace("\n", " ")
tree = ET.ElementTree(root)
tree.write(outputfile,'UTF-8',xml_declaration=True)