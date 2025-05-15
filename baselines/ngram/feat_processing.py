from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from os.path import join as join_path
import shutil
import xml.dom.minidom as md
import json
from lxml import html, etree
import random
import hashlib
#from IPython.core.debugger import set_trace
import re
from tqdm import tqdm

def clean_up_string(str):
    return re.sub(r'\s+', ' ', str).strip()

def get_html_url(json_file):
    try:
        with open(json_file) as inf:
            json_data = json.loads(inf.read())
            if json_data.get("html_content"):
                html = json_data["html_content"]
            elif json_data.get("html"):
                html = json_data["html"]
            elif json_data.get("content"):
                html = json_data["content"]
            else:
                print(f"Invalid json file content:{json_file}")
                return (None, None)
            
            url = json_data["url"].replace("\t", " ").replace("\n", " ")

            return (html.lower(), url.lower())
    
    except Exception as e:
        print("Invalid json file", json_file)
        return (None, None)

def create_feat_dict(input_path, dict_file, label_dict_file, min_feat_count):
    print(f"Creating feature dictionary from {input_path}")
    feature_dict = dict()
    label_dict = dict()
    labels = []
    with open(dict_file, "w") as outf:        
        for label in os.listdir(input_path):
            flist = os.listdir(f"{input_path}/{label}")
            label_dict[label] = len(labels)
            labels.append(label)
            print(f"Processing {label} samples.")
            for fn in tqdm(flist, total=len(flist)):
                json_file = f'{input_path}/{label}/{fn}'
                html, url = get_html_url(json_file)
                if html is None:
                    continue
                feats = get_feats(url, html)
                for feat in feats:
                    feature_dict[feat] = feature_dict.get(feat, 0) + 1
        
    reduced_dict = {key:feature_dict[key] for key in feature_dict if feature_dict[key] > min_feat_count}

    with open(dict_file, "w") as outf:
        for key in reduced_dict:
            outf.write(f"{key}\t{reduced_dict[key]}\n")

    with open(label_dict_file, "w") as outf:
        outf.write("\n".join(labels))

    feats =  list(reduced_dict.keys())
    return {feats[i]:i for i in range(len(feats))}, label_dict

def reduce_feat_file(input_feat, input_dict, output_feat, output_dict, max_feats):
    print(f"Reading feat dict file {input_dict}")
    with open(input_dict) as inf:
        i = 0
        feat_dict = []
        for line in inf:
            line = line.rstrip().split("\t")
            feat_dict.append((line[0], int(line[1]), i))
            i += 1
        
        feat_dict.sort(key=lambda x:-x[1])
    
    feat_dict = feat_dict[:max_feats]
    print(feat_dict[0])
    feat_dict_map = dict()
    i = 0
    for fd in feat_dict:
        feat_dict_map[fd[2]] = i
        i += 1

    with open(output_dict, "w") as outf:
        for fd in feat_dict:
            outf.write(f"{fd[0]}\t{fd[1]}\n")

    with open(input_feat) as inf, open(output_feat, "w") as outf:
        for line in inf:
            line = line.rstrip().split("\t")
            fvs = []
            for fv in line[1:]:
                j = int(fv.split(":")[0])
                if feat_dict_map.get(j):
                    fvs.append(feat_dict_map.get(j))
            
            if len(fvs) > 0:
                outf.write(line[0] + "\t" + "\t".join([f"{i}:1" for i in fvs]) + "\n")
    

def load_feat_dict(feature_dict_file, label_dict_file):
    feat_dict = dict()
    label_dict = dict()
    with open(feature_dict_file) as inf:
        i = 0
        for line in inf:
            line = line.split("\t")
            feat_dict[line[0]] = i
            i += 1

    with open(label_dict_file) as inf:
        i = 0
        for line in inf:
            label_dict[line.rstrip()] = i
            i += 1

    return feat_dict, label_dict

def create_libsvm_feat(input_path, feature_dict_file, label_dict_file, feat_file, train_mode):

    if train_mode:
        print(f"Creating feature dictionary file file from {input_path}")
        feature_dict, label_dict = create_feat_dict(input_path, feature_dict_file, label_dict_file, 10)
    else:
        print(f"Loading feature dictionary file {feature_dict_file}")
        feature_dict, label_dict = load_feat_dict(feature_dict_file=feature_dict_file, label_dict_file=label_dict_file)

    print(f"Creating LIBSVM file ")
    with open(feat_file, "w") as outf:
        for label in os.listdir(input_path):
            print(f"Processing {label} samples.") 
            if label_dict.get(label) is None:
                raise Exception(f"Invalid label {label}")
            
            flist = os.listdir(f"{input_path}/{label}")
            for fn in tqdm(flist, total=len(flist)):                
                json_file = f'{input_path}/{label}/{fn}'
                html, url = get_html_url(json_file)
                if html is None:
                    continue
                feat_index = dict()
                for feat in get_feats(url, html):
                    index = feature_dict.get(feat, -1)
                    if index >= 0:
                        feat_index[feat] = index

                fv = '\t'.join([f"{feat_index[feat]}:1" for feat in feat_index.keys()])
                outf.write(f"{label_dict.get(label)}\t{fv}\n")
    
def get_xpaths_ngrams(html, ngram=4):

    htmlparser = etree.HTMLParser()
    root = etree.fromstring(html, htmlparser)
    tree = etree.ElementTree(root)

    #root = etree.fromstring(html_content)    
    #tree = etree.ElementTree(root)
    xpaths = []    
    xpaths_lengths = []    
    for element in root.iter():
        try:    
            xpath = tree.getpath(element).split("/")
            xpaths_lengths.append(len(xpath))
        except:
            continue
        
        if len(xpath) < ngram:
            xpaths.append("/".join(xpath))
            continue

        for i in range(len(xpath) - ngram):
            xpaths.append("/".join(xpath[i:i+ngram]))

    return list(set(xpaths)), max(xpaths_lengths)

def get_feats(url, html):
    html = html.lower()
    soup = BeautifulSoup(html, 'html.parser')

    all_text = soup.get_text()

    all_scripts = []
    all_src_values = []
    for script in soup.find_all('script'):
        all_scripts.append(script.get_text())
        src_value = script.get('src')
        if src_value:
            all_src_values.append(src_value)

    all_link_values = [link.get('href') for link in soup.find_all('link') if link.get('href')]

    img_data_src_values = []
    img_src_values = [] 
    for img in soup.find_all('img'):
        for attr_name in img.attrs:
            if attr_name.startswith("data-"):
                img_data_src_values.append(img.get(attr_name))        
            if attr_name.startswith("src"):
                img_src_values.append(img.get(attr_name))        

    all_tags = [tag.name for tag in soup.find_all()]

    try:
        xpaths, xpaths_max_length = get_xpaths_ngrams(html)
    except Exception as e:
        print(f"Error in getting the xpaths: {e}, {url}")
        xpaths = ["body"]

    sep = " "

    proto_feats = {
            "te": clean_up_string(all_text),
            "ta": clean_up_string(sep.join(all_tags)),
            "sc": clean_up_string(sep.join(all_scripts)), 
            "ss": clean_up_string(sep.join(all_src_values)), 
            "li": clean_up_string(sep.join(all_link_values)), 
            "is": clean_up_string(sep.join(img_src_values)),
            "id": clean_up_string(sep.join(img_data_src_values))
    }

    ngram_feats = make_ngrams(proto_feats, ngram_length=4)
    return ngram_feats + xpaths

def make_ngrams(proto_feats, ngram_length):
    ngrams = []
    for name in proto_feats:
        cur_ngrams = []
        feats = proto_feats[name]
        for i in range(len(feats) - ngram_length):
            cur_ngrams.append(f"{name}_{feats[i:i+ngram_length]}")

        ngrams += cur_ngrams
    
    return ngrams

