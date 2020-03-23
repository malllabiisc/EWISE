import argparse
import os

def xml_reader(gold_file, xml_file, output_filename):
    import xml.etree.ElementTree as ET
    import sys
    import json
    from nltk.corpus import wordnet as wn

    d = {}
    
    with open(gold_file,'r') as map_file:
        for line in map_file:
            d[line.strip().split()[0]] = line.strip().split()[1]

    tree = ET.parse(xml_file)
    root = tree.getroot()
    final_json = []
    for doc in root:
        for sent in doc:
            count = 0
            l = []
            orig = ""
            anno = ""
            stem = []
            pos = []
            for tok in sent:
                try:
                    att = tok.attrib['id']
                    orig = orig + " " + "_".join(tok.text.strip().split()) # hyphen or underscore?
                    anno = anno + " " + d[tok.attrib['id']]
                    l.append(count)
                    count += 1
                    stem.append(tok.attrib['lemma'])
                    pos.append(tok.attrib['pos'])
                except:
                    orig = orig + " " + tok.text
                    anno = anno + " " + tok.text
                    count += 1
            final_json.append({
                'original': orig.strip().lower(),
                'annotated' : anno.strip().lower(),
                'offsets' : l,
                'doc_offset' : sent.attrib["id"],
                'stems': stem,
                'pos': pos
                })
    op_fname = str(output_filename)+".json"
    with open(op_fname, 'w') as outfile:
        json.dump(final_json, outfile)

if(__name__=='__main__'):
    parser = argparse.ArgumentParser(description='reads xml and converts it to json with the dictionary information incorporated in it.')
    parser.add_argument('--goldfile', type=str,
                        help='gold file name')
    parser.add_argument('--xmlfile', type=str,
                        help='xml file name')
    parser.add_argument('--opname', type=str,
                        help='output file name')
    parser.add_argument('--opdir', type=str, default='./temp',
                        help='output dir path')
    args = parser.parse_args()
    #print (args)                    
    xml_reader(args.goldfile, args.xmlfile, os.path.join(args.opdir, args.opname))
