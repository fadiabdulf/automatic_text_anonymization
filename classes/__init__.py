import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BS
# import logging
from collections import OrderedDict
import os
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import ast
from nltk.corpus import stopwords 
import string
stop_words = set(stopwords.words('english')) 
try:
    from glove import Corpus, Glove
except: 
    pass


# logging.basicConfig(format="%(asctime)s : %(thread)d : %(levelname)s : %(message)s", level=logging.INFO)

class MyIterator:
    def __init__(self, docs):
        self.docs = docs
        
    def __iter__(self):
        for d in tqdm(self.docs):
            yield d

class DataLoader(object):
    """
    Data loader
    """
    def __init__(self, input_dir, output_dir):
        """
        input_dir: the input directory.
        output_dir: the output directory.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.soups = None
        self.lines = None
        self.chunk_lbl = None

    def load(self, load_xml=None, generate=False):
        """
        load_xml: load xml file from the directory 'abstract_xml/'.
        """
        soups = OrderedDict()
        csv_file = 'filter.csv'
        if load_xml is None:
            in_dir = self.input_dir
            xml_path = self.input_dir
        elif not load_xml:
            in_dir = self.input_dir + 'actors/'
            xml_path = self.input_dir + 'abstract_xml/'
            csv = pd.read_csv(in_dir + csv_file)
        else:
            in_dir = self.input_dir + 'abstract_xml/'
            xml_path = self.input_dir + 'abstract_xml/'
        i = 0
        # r=root, d=directories, f = files
        for r, d, f in os.walk(in_dir):
            #print(f)
            for file in tqdm(f):
                if not (file.lower().endswith('.txt') or file.lower().endswith('.xml')):
                    continue
                #if i % 1000:
                #    logging.info("read {0} ".format(file))
                if load_xml or load_xml is None:
                    if file.startswith('actor_7616'):
                        soups[file] = self.read_soup(r, file, 'html.parser')
                    else:
                        soups[file] = self.read_soup(r, file, 'xml')
                    if generate:
                        name = soups[file].find('title').text
                        link = soups[file].find('link').text
                        tags = soups[file].find('tags')
                        annotated = soups[file].find('annotated').text
                        abst = soups[file].find('abstract').text
                        abst = str(abst).replace('&amp;', '&')
                        text = soups[file].find('originalabstract').text
                        text = str(text).replace('&amp;', '&')
                        tokens = soups[file].find('stanford_tokens').text
                        chunks = soups[file].find('opennlp_chunks').text
                        microlabels = soups[file].find('microlabels1').text if annotated == '1' else ""
                        abstractpos = soups[file].find('stanford_pos').text
                        temp = self.create_soup(name, link, annotated, text, abst, tokens, chunks, abstractpos, microlabels, tags)
                        if annotated == '1':
                            for tag in ['nerText3', 'nerText4', 'nerText7']:
                                temp.find(tag.lower()).string = soups[file].find(tag).text
                        soups[file] = temp

                elif file.lower().endswith('.txt'):
                    file_path = r + file
                    with open(file_path, 'rb') as f:
                        blines = f.readlines()
                    lines = [bline.decode("utf-8") for bline in blines]
                    text = ''.join(lines)
                    text = str(text).replace('&amp;', '&')
                    params = csv.iloc[int(file.split('_')[1][:-4])]
                    tags = '<tags></tags>'
                    annotated = '0'
                    if params['filter'] == 'TRUE':
                        soup = self.read_soup(xml_path, file.replace('.txt', '.xml'))
                        tags = soup.find('tags')
                        annotated = soup.find('annotated').text
                        text = soup.find('originalabstract').text
                        text = str(text).replace('&amp;', '&')
                    soups[file.lower().replace('.txt', '.xml')] = self.create_soup(params['name'], params['link'], annotated, text, tags)
        self.soups = soups

    def get_soups(self):
        if self.soups is None:
            self.load()
        return self.soups

        
    def get_lines(self, chunk=True, refresh=False, abs_tag='originalabstract', lower=True, test_only=False, tokenizer='stanford', testTokenizer=False, originalChunk=False):
        if self.lines is None or refresh == True:
            self.lines = OrderedDict()
            #logging.info("reading all the abstracts ... this may take a while")
            i = 0
            soups = self.get_soups()
            for key in tqdm(soups):
                if test_only and soups[key].find('annotated').text == '0':
                    continue
                #if (i % 1000 == 0):
                #    title = soups[key].find('title').text.replace(' ', '_')
                #    logging.info("read {0} abstract for the actor: {1}".format(i, title))
                #print(soups[key])
                abstract = soups[key].find(abs_tag).text
                annotated = soups[key].find('annotated').text == '1'
                chunks = []
                try:
                    if chunk:
                        if tokenizer == 'stanford' and not (testTokenizer and annotated):
                            chunks = ast.literal_eval(soups[key].find('opennlp_chunks').text)
                        else:
                            chunks = ast.literal_eval(soups[key].find('spacy_chunks').text) 
                except:
                    chunks = []
                try:
                    if tokenizer == 'stanford' and not (testTokenizer and annotated):
                        tokens = ast.literal_eval(soups[key].find('stanford_tokens').text[len('OrderedDict('):-1])
                    else:
                        tokens = ast.literal_eval(soups[key].find('spacy_tokens').text[len('OrderedDict('):-1]) 
                        
                    if chunk:
                        chunks = [('_'.join(c[0].split()), c[1], c[2]) for c in chunks]
                        line =  ' '.join([w for w, i in self.apply_chunk(abstract, tokens, chunks, originalChunk=originalChunk)])
                    else:
                        line = ' '.join([w for i, w in tokens])
                    if lower:
                        line = line.lower()
                    words = [w for w in line.split(' ') if not w in stop_words and w not in list(string.punctuation)]
                except:
                    #print(key)
                    words = ['none']
                i = i + 1
                self.lines[key] = words
        return self.lines

    def apply_chunk(self, text, tokens, chunks, rep='_', originalChunk=False):
        i = 0
        chunk = chunks[0] if len(chunks) > 0 else ('', -1, -1) 
        res = []
        in_chunk = False
        r = []
        for idx, token in tokens:
            if in_chunk:
                res[-1] = (res[-1][0] + rep + token, res[-1][1] + 1) if not originalChunk else  (chunk[0], res[-1][1] + 1)
            #here
            #if idx <= chunk[1] and idx + len(token) >= chunk[1]:
            if idx == chunk[1]:
                res.append((token, 1) if not originalChunk else  (chunk[0], 1))
                in_chunk = True
            elif not in_chunk:
                res.append((token, 1))
            
            if idx + len(token) >= chunk[2]:
                i = i + 1
                chunk = chunks[i] if i < len(chunks) else ('', -1, -1)
                in_chunk = False
        return res

    def get_label(self, index, value, all_tags):
        #if len(value) >= len('impressionist_,_screenwriter_,_musician_,_producer_and_painter'):
        #    print(value)
        rang = index + len(value)
        lbl = 'O'
        #logging.info(str(index) + "," + str(value) + "," + str(i))
        for start, length, tag in all_tags:
            if (start >= index and start + length < rang) or (start < rang and start + length >= rang):
                lbl = tag
                break
        return lbl
    
    def get_chunks(self, soup, abs_tag='originalabstract', tokenizer='stanford', testTokenizer=False, originalChunk=False):
        abstract = soup.find(abs_tag).text
        annotated = abstract = soup.find('annotated').text == '1'
        if tokenizer == 'stanford' and not (testTokenizer and annotated):
            chunks = ast.literal_eval(soup.find('opennlp_chunks').text)
            tokens = ast.literal_eval(soup.find('stanford_tokens').text[len('OrderedDict('):-1])
        else:
            chunks = ast.literal_eval(soup.find('spacy_chunks').text)
            tokens = ast.literal_eval(soup.find('spacy_tokens').text[len('OrderedDict('):-1])
        chunks = [('_'.join(c[0].split()), c[1], c[2]) for c in chunks]
        res = self.apply_chunk(abstract, tokens, chunks, originalChunk=originalChunk)
        return res
        

    def get_labels(self, soup, tokens=None, lower=False, tokenizer='stanford', testTokenizer=False, originalChunk=False):
        if tokens is None:
            tokens = self.get_chunks(soup, tokenizer=tokenizer, testTokenizer=testTokenizer, originalChunk=originalChunk)
            #tokens = ast.literal_eval(soup.find('spacy_tokens').text[len('OrderedDict('):-1])
        annotated = soup.find('annotated').text == '1'
        if tokenizer == 'stanford' and not (testTokenizer and annotated):
            lbls = ast.literal_eval(soup.find('microlabels1').text)
            postags = ast.literal_eval(soup.find('stanford_pos').text) 
        else:
            lbls = ast.literal_eval(soup.find('microlabels2').text)
            postags = ast.literal_eval(soup.find('spacy_pos').text) 
        labeled_data = []
        i = 0
        for token in tokens:
            lbl = ""
            pos = ""
            chunk_len = token[1]
            lbl = ''.join(lbls[i:i+chunk_len])
            pos = '_'.join(postags[i:i+chunk_len])
            if lower:
                labeled_data.append((token[0].lower(), lbl, pos))
            else:
                labeled_data.append((token[0], lbl, pos))
            i = i + chunk_len
        return labeled_data

    def load_lbl(self, soups, lower=True, chunking=True, tokenizer='stanford', testTokenizer=False, originalChunk=False):
        """This method reads the input file text and merge them with the labels"""
        #logging.info("reading file {0}...this may take a while".format(input_file))
        for key in tqdm(soups):
            soup = soups[key]
            title = soup.find('title').text.replace(' ', '_')
            abstract = soup.find('originalabstract').text
            annotated = soup.find('annotated').text == '1'
            #if (i % 1000 == 0 and not testSet):
            #    logging.info("read {0} reviews for the actor: {1}".format(i, title))
            # do some pre-processing and return list of words for each review
            # text
            if chunking:
                words = self.get_chunks(soups[key], tokenizer=tokenizer, originalChunk=originalChunk)
            else:
                if tokenizer == 'stanford' and not (testTokenizer and annotated):
                    words =  [(w, 1) for i, w in ast.literal_eval(soup.find('stanford_tokens').text[len('OrderedDict('):-1])]
                else:
                    words =  [(w, 1) for i, w in ast.literal_eval(soup.find('spacy_tokens').text[len('OrderedDict('):-1])]
            #print(words)
            #logging.info("{0}: read {1} review for the actor: {2}".format(j, key, title))
            text = abstract.lower()
            yield key , self.get_labels(soup, words, lower=lower, tokenizer=tokenizer, testTokenizer=testTokenizer)

    def get_chunk_lbl(self, lower=True, chunking=True, refresh=False, tokenizer='stanford', testTokenizer=False, originalChunk=False):
        if self.chunk_lbl is None or refresh:
            soups = {key: soup for key, soup in self.get_soups().items() if soup.find('annotated').text == '1'}
            self.chunk_lbl = OrderedDict({key: doc for key, doc in self.load_lbl(soups, lower=lower, chunking=chunking, tokenizer=tokenizer, testTokenizer=testTokenizer, originalChunk=originalChunk)})
        return self.chunk_lbl

    def create_soup(self, name, link, annotated, text, abst, tokens, chunks, abstractpos, microlabels, tags):
        xml = """<doc>
<title>%s</title>

<link>%s</link>

<annotated>%s</annotated>

<originalabstract>%s</originalabstract>

<abstract>%s</abstract>

<stanford_tokens>%s</stanford_tokens>

<opennlp_chunks>%s</opennlp_chunks>

<stanford_pos>%s</stanford_pos>

<microlabels1>%s</microlabels1>

<spacy_tokens></spacy_tokens>

<spacy_chunks></spacy_chunks>

<spacy_pos></spacy_pos>

<microlabels2></microlabels2>

<nertext3></nertext3>

<nertext4></nertext4>

<nertext7></nertext7>

<word2vec></word2vec>

<word2vec_gen></word2vec_gen>

%s
</doc>"""
        #xml = "<doc>\n<title>%s</title>\n<link>%s</link>\n<annotated>%s</annotated>\n<originalAbstract>%s</originalAbstract>\n<tokens></tokens>\n<abstractpos></abstractpos>\n<microlabels></microlabels>\n<word2vec_gen></word2vec_gen>\n%s\n</doc>"
        soup = BS(xml % (name, link, annotated, '<![CDATA[' + text + ']]>', '<![CDATA[' + abst + ']]>', tokens, chunks, abstractpos, microlabels, tags), 'xml')
        return soup

    def read_soup(self, path, file, parser='html.parser'):
        file_path = path + file
        with open(file_path, 'rb') as f:
            blines = f.readlines()
        lines = [bline.decode("utf-8") for bline in blines]
        text = ''.join(lines)
        soup = BS(text, parser)
        #soup = BS(text, 'lxml-xml')
        return soup

    def pre_proc_tag(self, tag):
        if tag.name != 'deidi2b2':
            tag.name = tag.name.upper()
        for t in tag.findChildren():
            self.pre_proc_tag(t)
        return tag

    def write_soup(self, path, file, soup, prettify=False):
        #pre_proc_tag(soup)
        #soup.find('doc').name = 'doc'
        file_path = path + file
        with open(file_path, 'wb') as f:
            if not prettify:
                f.write(str(soup).encode('utf-8'))
            else:
                f.write(str(soup.prettify()).encode('utf-8'))
        return soup


import spacy
import requests
import re
#from tqdm import tqdm_notebook as tqdm
import threading
import time

class myThread(threading.Thread):
    def __init__(self, threadID, name, soups, chunker):
        threading.Thread.__init__(self)
        self.chunker = chunker
        self.threadID = threadID
        self.name = name
        self.soups = soups
        self.results = {}

    def run(self):
        #logging.info("Starting " + self.name)
        self.results = self.chunker.chunking(self.soups)
        #logging.info("Exiting " + self.name)


class Chunker(object):
    def __init__(self):
        self.months = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
        self.date_format1 = '((?:19|20)\d\d)[- /.](0?[1-9]|1[012])[- /.](0?[1-9]|[12][0-9]|3[01])|((0?[1-9]|[12][0-9]|3[01])[- /.](0?[1-9]|1[012])[- /.](?:19|20)\d\d)'
        self.date_format2 = '(({0})(\s|,)+\d+(\s|,)*\d{{4}}|\d+(\s|,)*({0})(\s|,)*\d{{4}})'.format('|'.join(self.months))
        self.rules = [re.compile(self.date_format1, re.IGNORECASE), re.compile(self.date_format2, re.IGNORECASE)]
        self.nlp = None
        self.cloned_soups = None

    def get_chunked_soups(self, soups, slice_size=300, out_dir=None, refresh=False):
        if self.nlp is None:
            print("chunking all the abstracts ... this may take a while")
            self.nlp = spacy.load("en_core_web_md")

        if self.cloned_soups is None or refresh:
            slices = []
            i = 0
            for key in soups:
                if i % slice_size == 0:
                    slices.append({})
                #print(i, int(i % slice_size), slices)
                slices[int(i / slice_size)][key] = soups[key]
                i = i + 1

            threads = [myThread(i, "Thread-" + str(i), slices[i], self) for i in range(len(slices))]

            for thread in threads:
                thread.start()
                time.sleep(15.0)

            for thread in threads:
                thread.join()
            
            new_soups = {}
            for thread in threads:
                new_soups.update(thread.results)

            self.cloned_soups = new_soups

            if not out_dir is None:
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)

                for key in new_soups:
                    #print(new_soups[key].find('doc'))
                    self.write_soup(out_dir, key, new_soups[key])

            print ("Exiting Main Thread")

        return self.cloned_soups

    def fix_tokens(self, chunks, tokens, pos):
        i = 0
        for k in range(len(chunks)):
            chunk, start, end = chunks[k]
            while tokens[i][0] + len(tokens[i][1]) < end:
                i += 1
            if tokens[i][0] + len(tokens[i][1]) == end:
                pass
            else:
                tokens.insert(i+1, (end - tokens[i][0], tokens[i][1][end - tokens[i][0]:]))
                tokens[i] = (tokens[i][0], tokens[i][1][:end - tokens[i][0]])
                pos.insert(i+1, pos[i])
                #print('E:%s, %s' % (chunk +, tokens[i]))
            
        return tokens, pos

    def chunking(self, soups):
        i = 0
        cloned_soups = {}
        for key in tqdm(soups):
            soup = soups[key]
            title = soup.find('title').text.replace(' ', '_')
            #if (i % 1000 == 0):
            #    logging.info("read {0} abstract for the actor: {1}".format(i, title))
            try:
                m = self.wiki_chunking(soup)
            except:
                m = set()
                #logging.info("{0}: Error for the actor {1}".format(i, title))
            mm, doc = self.spaCy_chunking(soup)
            chunks = self.merge_chunks(m, mm, soup)
            cloned = BS(str(soup), 'html.parser')
            #cloned.find('spacy_tokens').string = str(OrderedDict([(t.idx,t.text) for t in doc if t.tag_ != '_SP']))
            
            if soup.find('annotated').text == '1':
                tags = self.get_tags_start_end(soup)
                lbl = self.get_lbl_string(tags, doc)
                cloned.find('microlabels2').string = str(lbl)
            #tag = cloned.new_tag('chunks')
            #tag.string = str(sorted(chunks, key=lambda x: x[1]))
            #cloned.doc.insert(-2, tag)
            #old_chunks = ast.literal_eval(soup.find('chunks').text)
            chunks = sorted(chunks, key=lambda x: x[1])
            pos = [t.tag_ for t in doc if t.tag_ != '_SP']
            tokens = [(t.idx,t.text) for t in doc if t.tag_ != '_SP']
            cloned.find('spacy_chunks').string = str(chunks)
            #tokens, pos = self.fix_tokens(chunks, tokens, pos)
            cloned.find('spacy_tokens').string = str(OrderedDict(tokens))
            cloned.find('spacy_pos').string = str(pos)
            #chunks[0] = old_chunks[0]
            cloned_soups[key] = cloned
            i = i + 1
        return cloned_soups

    def write_soup(self, path, file, soup, prettify=False):
        #pre_proc_tag(soup)
        #soup.find('doc').name = 'doc'
        file_path = path + file
        with open(file_path, 'wb') as f:
            if not prettify:
                f.write(str(soup).encode('utf-8'))
            else:
                f.write(str(soup.prettify()).encode('utf-8'))
        return soup
        


    def extend_links(self, links, sort=True):
        extended = []
        for e in links:
            es = e.split(None)
            extended.extend([' '.join(es[:i]) for i in range(len(es), max(len(es) - 3, 0), -1)])
        if sort:
            return sorted(set(extended), key=lambda e: len(e), reverse=True)
        return extended

    def get_wiki_links(self, title):
        URL = 'https://en.wikipedia.org/w/api.php'
        PARAMS = {'format':'json', 'action':'query', 'prop':'links', 
                'redirects':'1','titles':title, 'pllimit':'max'}

        content = requests.get(url = URL, params = PARAMS).json()
        all_links = [link['title'] for link in content['query']['pages'][list(content['query']['pages'])[0]]['links']]
        while 'continue' in content: 
            if 'plcontinue' in content['continue']:
                PARAMS['plcontinue'] = content['continue']['plcontinue']
            else:
                PARAMS['plcontinue'] = content['continue']['elcontinue']
            content = requests.get(url = URL, params = PARAMS).json()
            links = [link['title'] for link in content['query']['pages'][list(content['query']['pages'])[0]]['links']]
            all_links.extend(links)
        links = all_links
        return links
    
    def match(self, text, word):
        ind = text.lower().find(word.lower())
        if (len(word) >= 6 or len(word.split(None)) > 1) and ind != -1:
            return (word, ind, ind + len(word))
        return None

    def pre_matched(self, link, pre):
        for lnk in pre:
            if lnk.lower().startswith(link.lower()):
                return True
        return False

    def intersect(self, matches, m):
        for match in matches:
            if m[1] >= match[1] and m[2] <= match[2]:
                return match
        return None

    def left_intersect(self, matches, m):
        for match in matches:
            if m[1] < match[1] and m[2] >= match[1] and m[2] <= match[2]:
                return match
        return None

    def right_intersect(self, matches, m):
        for match in matches:
            if m[1] >= match[1] and m[1] <= match[2] and m[2] > match[2]:
                return match
        return None
    
    def reg_chunking(self, text):
        chunks = []
        for rule in self.rules:
            pos = 0
            while True:
                m = rule.search(text, pos)
                if m:
                    pos = m.span()[1]
                    chunks.append((m.group(), m.span()[0], m.span()[1]))
                else:
                    break
        return chunks

    def wiki_chunking(self, soup, reg_chunk=True):
        title = soup.find('title').text
        text = soup.find('originalabstract').text
        matches = set(self.reg_chunking(text))
        pre = []
        #print(title)
        links = self.get_wiki_links(title)
        links = self.extend_links(links)
        for link in links:
            if not self.pre_matched(link, pre):
                m = self.match(text, link)
                if m:
                    pre.append(link)
                    if self.intersect(matches, m) is None:
                        matches.add(m)
        return matches

    def spaCy_chunking(self, soup):
        title = soup.find('title').text
        text = soup.find('originalabstract').text
        matches = set()
        #print(title)
        doc = self.nlp(text)
        for chunk in doc.noun_chunks:
            m = self.match(text, chunk.text)
            if m:
                matches.add(m)
        return matches, doc

    def merge_chunks(self, chunks1, chunks2, soup):
        chunks3 = set()
        text = soup.find('originalabstract').text
        title = soup.find('title').text
        #print(title)
        for chunk in sorted(chunks1 | chunks2, key=lambda x: len(x[0]), reverse=True):
            if self.intersect(chunks3, chunk) is None:
                chunks3.add(chunk)
                left = None
                right = None
            else:
                left = self.left_intersect(chunks3, chunk)
                right = self.right_intersect(chunks3, chunk)
            if left and right:
                chunks3.remove(left)
                chunks3.remove(right)
                new_chunk = (text[left[1]:right[2]], left[1], right[2])
                chunks3.add(new_chunk)
            elif left:
                chunks3.remove(left)
                new_chunk = (text[left[1]:chunk[2]], left[1], chunk[2])
                chunks3.add(new_chunk)
            elif right:
                chunks3.remove(right)
                new_chunk = (text[chunk[1]:right[2]], chunk[1], right[2])
                chunks3.add(new_chunk)
        return chunks3

    def get_tags_start_end(self, soup):
        res = []
        for tag in soup.find('tags').findChildren():
            res.append((tag['value'], int(tag['offset']), int(tag['offset']) + int(tag['length'])))
        return res

    def get_lbl_string(self, tags, doc):
        res = []
        tag_id = 0
        for t in doc:
            if t.tag_ == '_SP':
                continue
            if tag_id >= len(tags):
                res.append('O')
                continue
            if t.idx <= tags[tag_id][1] and t.idx + len(t) >= tags[tag_id][1]:
                res.append('B')
            elif t.idx >= tags[tag_id][1] and t.idx <= tags[tag_id][2]:
                res.append('I')
            else:
                res.append('O')
            
            if t.idx + len(t) >= tags[tag_id][2]:
                tag_id = tag_id + 1
        return res
        

import pickle
import gensim
from gensim.models import Word2Vec, FastText

class WordEmbedding(object):

    def __init__(self, sg, size, window, min_count, t='word2vec', workers=10, hashfxn=None):
        self.sg = sg
        self.vec_size = size
        self.window = window
        self.min_count = min_count
        self.t = t
        self.workers = workers
        if hashfxn is None:
            self.hashfxn = self.hash
        else:
            self.hashfxn = hashfxn

    def hash(self, astring):
        return ord(astring[0])

    def fit(self, X, epochs=10):
        self.lines = X
        if self.t == 'word2vec':
            #build vocabulary and train model
            self.model = Word2Vec(
                    self.lines,
                    sg=self.sg,
                    size=self.vec_size,
                    window=self.window,
                    min_count=self.min_count,
                    workers=self.workers,
                    hashfxn=self.hashfxn)
        elif self.t == 'fasttext':
            self.model = FastText(
                    self.lines,
                    sg=self.sg,
                    size=self.vec_size,
                    window=self.window,
                    min_count=self.min_count,
                    workers=self.workers,
                    hashfxn=self.hashfxn)
        elif self.t == 'glove':
            #Creating a corpus object
            corpus = Corpus() 
            #Training the corpus to generate the co occurence matrix which is used in GloVe
            corpus.fit(self.lines, window=self.window)

            glove = Glove(no_components=self.vec_size, learning_rate=0.05) 
            glove.fit(corpus.matrix, epochs=epochs, no_threads=self.workers, verbose=True)
            glove.add_dictionary(corpus.dictionary)
            with open('./models/glove/glove.txt', 'w+') as f:
                for word, vec in zip(corpus.dictionary, glove.word_vectors):
                    f.write(' '.join([word] + list(map(str, vec))))
                    f.write('\n')
            self.model = self.load_glove('./models/glove/glove.txt')
            
        print('Start training:')
        if self.t in ['word2vec', 'fasttext']:
            self.model.train(self.lines, total_examples=len(list(self.lines)), epochs=epochs)

    def load_glove(self, model_path):
        from gensim.test.utils import datapath, get_tmpfile
        from gensim.models import KeyedVectors
        from gensim.scripts.glove2word2vec import glove2word2vec
        
        glove_file = datapath(os.path.abspath(model_path))
        tmp_file = get_tmpfile("test_word2vec.txt")

        _ = glove2word2vec(glove_file, tmp_file)

        model = KeyedVectors.load_word2vec_format(tmp_file)
        return model
    
    def update(self, new_X, epochs):
        self.model.build_vocab(new_X, update=True)
        print("start update")
        self.model.train(new_X, total_examples=len(new_X), epochs=epochs)

    def save(self, filename=None):
        if not os.path.isdir('./models'):
            os.mkdir('./models')
        if filename is None:
            filename = "models/{0}1_{1}_{2}_{3}_{4}.model".format(self.t, self.vec_size, self.window, self.sg, self.min_count)
        self._save_object(self.model, filename)

    def load(self, filename=None, model_type='word2vec'):
        if filename is None:
            filename = "models/{0}1_{1}_{2}_{3}_{4}.model".format(self.t, self.vec_size, self.window, self.sg, self.min_count)
            self.model = self._load_object(filename)
        else:
            if model_type == 'word2vec':
                self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
            elif model_type == 'fasttext':
                self.model = gensim.models.FastText.load_fasttext_format(filename, encoding='utf8')
            elif model_type == 'glove':
                self.model = self.load_glove(filename)
            else:
                pass

    
    def _save_object(self, obj, filename):
        #logging.info("save object started ...")
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        #logging.info("save object finished ...")
            
    def _load_object(self, filename):
        #logging.info("load object %s started ..." % filename)
        obj = None
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
        #logging.info("load object finished ...")
        return obj


# generalize class
import wikipedia
import requests 
import re
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup as BS

class Generlaizer(object):
    def __init__(self):
        self.months = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']

        self.d_e = '^((?:19|20)\d\d)[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$'

        #professions = ['actor', 'actress', 'director', 'singer', 'model', 'producer']

        #award_words = ['award', 'nomination', 'nominate']

        #movie_words = ['film', 'movie', 'comedy', 'comedic', 'trilogy', 'drama', 'cinematic', 'series', 'tv']

        puncs = list(string.punctuation)
        #print(puncs)
        self.puncs = [re.escape(p) for p in puncs]

        # classes = {'r01': 'DATE', 'r02': 'MOVIE', 'r03': 'YEAR', 'r04': 'PROFESSION', 'r05': 'AWARD', 'r06': 'MOVIE',
        #            'r07': 'PUNCT', 'r08': 'NUMBER', 'r09': 'STOPWORD'}
        self.classes = {'r01': 'DATE', 'r02': 'MOVIE', 'r03': 'YEAR', 'r06': 'MOVIE',
                        'r07': 'PUNCT', 'r08': 'NUMBER', 'r09': 'STOPWORD'}

        r = {}
        r['r01'] = re.compile('.*{0}.*'.format('|'.join(self.months)), re.IGNORECASE)
        #r['r02'] = re.compile('.*_\(_\d{4}_?\)?', re.IGNORECASE)
        r['r03'] = re.compile('^[1-2]\d{3}$', re.IGNORECASE)
        # r['r04'] = re.compile('.*{0}.*'.format('|'.join(professions)), re.IGNORECASE)
        # r['r05'] = re.compile('.*{0}.*'.format('|'.join(award_words)), re.IGNORECASE)
        #r['r06'] = re.compile('.*{0}.*'.format('|'.join(movie_words)), re.IGNORECASE)
        r['r07'] = re.compile('^({0})$'.format('|'.join(self.puncs)), re.IGNORECASE)
        r['r08'] = re.compile('^\d+$', re.IGNORECASE)
        r['r09'] = re.compile('^({0})$'.format('|'.join(stop_words)), re.IGNORECASE)
        self.r = r

    def classification(self, df):
        df1 = df.copy()
        #df1['class'] = df1.apply(lambda _: 'NONE', axis=1)
        sorted_rules = sorted(list(self.r))
        for ind, row in df1.iterrows():
            for key in sorted_rules:
                rule = self.r[key]
                if rule.search(row['term']):
                    df1.at[ind, 'class'] = self.classes[key]
                    break
        return df1

    def generalize(self, df):
        df1 = df.copy()
        df1['generalizations'] = 'NONE'
        for ind, row in df1.iterrows():
            text = row['term']
            res = []
            try:
                while True:
                    pos = text.index('_')
                    res.append(text[pos + 1:])
                    text = text[pos + 1:]
            except Exception as e:
                #print(e)
                pass
            df1.at[ind, 'generalizations'] = res
        return df1


    def check_info(self, info, a):
        for i in info:
            if i in a.text:
                return True
        return False

    def get_a_value(self, a, prop):
        if prop == 'TEXT':
            return a.text
        else:
            return a.text[1:-1]
    
    def get_entity(self, term, URL, content, info = ['hasWikipediaUrl', 'owl:sameAs', 'rdf:type'], props = ['xlink:href', 'xlink:href', 'TEXT'],
                repeat=False, page_id=None):
        #print(content[:10])
        soup = BS(content, 'html.parser')
        svg = soup.find('svg')
        props_dict = {elem:prob for elem, prob in zip(info, props)}
        j = 0
        get_info = False
        if not repeat:
            res = {}
        else:
            res = []
        for a in svg.findChildren('a'):
            if get_info:
                elem = info[j]
                v = self.get_a_value(a, props_dict[elem])
                
                if not repeat:
                    j = j + 1
                    res[elem] = v
                else:
                    res.append(v)
            if self.check_info(info, a):
                get_info = True
                continue
            else:
                get_info = False
    
        if not page_id is None and len(res) > 0: 
            PARAMS = {'entity':'<%s>' % term,'relation':'rdf:type', 'page': page_id}
            content = requests.get(url = URL, params = PARAMS).content
            res2 = self.get_entity(term, URL, content, info=info, props=props, repeat=repeat, page_id=page_id+1)
            res.extend(res2)
        
        return res

    def check_name(self, term):
        # api-endpoint 
        URL = "https://gate.d5.mpi-inf.mpg.de/webyago3spotlx/EntityServlet"
        # defining a params dict for the parameters to be sent to the API 
        PARAMS = {'term' : term, 'table' : 'yagopreflabels'}
        # sending get request and saving the response as response object 
        r = requests.get(url = URL, params = PARAMS) 
        # extracting data in json format 
        data = r.json() 
        term = '<%s>' % term
        for t in data:
            if t.lower() == term.lower():
                return t, True
        return term, False

    def get_from_max(self, term, check=False):
        if self.check_name:
            term, checked = self.check_name(term)
        # api-endpoint 
        URL = "https://gate.d5.mpi-inf.mpg.de/webyago3spotlx/SvgBrowser"
        #"https://gate.d5.mpi-inf.mpg.de/webyago3spotlx/SvgBrowser?entity=%3CRobin_Tunney%3E&relation=rdf%3Atype"
        # defining a params dict for the parameters to be sent to the API 
        PARAMS = {'entityIn' : term, 'codeIn': 'eng', 'table' : 'yagopreflabels'}
        # sending get request and saving the response as response object 
        r = requests.get(url = URL, params = PARAMS) 
        res1 = self.get_entity(term, URL, r.content)
        PARAMS = {'entity':'<%s>' % term if not check else term,'relation':'rdf:type'}
        res2 = self.get_entity(term, URL, requests.get(url = URL, params = PARAMS).content, info=['rdf:type'], 
                                                props=['TEXT'], repeat=True, page_id=0)
        return res1, res2

    def is_it_same(self, term, title):
        terms = set([w for w in term.lower().split('_') if not w in self.puncs])
        tw = set([w for w in title.lower().split() if not w in self.puncs])
        return len(terms.intersection(tw)) > 0

    def generalize2(self, term, gen):
        res = {}
        res['wiki'] = []
        res['dbpedia'] = []
        res['rdf_type'] = []
        res['generalizations2'] = []
        sub_terms = [term] + gen
        try:
            wiki = []
            dbpedia = []
            rdf_type = []
            generalizations2 = []
            for term in sub_terms:
                t1, t2 = self.get_from_max(term, check=True)
                if len(t1) > 0:
                    if 'hasWikipediaUrl' in t1:
                        wiki.append(t1['hasWikipediaUrl'])
                    if 'owl:sameAs' in t1:
                        dbpedia.append(t1['owl:sameAs'])
                    if 'rdf:type' in t1:
                        rdf_type.append(t1['rdf:type'])
                    generalizations2.extend(t2)
            res['wiki'] = wiki
            res['dbpedia'] = dbpedia
            res['rdf_type'] = rdf_type
            res['generalizations2'] = generalizations2
        except:
            pass
        return res

    def generalize3(self, term):
        res = []
        try:
            page_title = wikipedia.page(term).title
            if self.is_it_same(term, page_title):
                categories = wikipedia.page(term).categories        
                res = categories
            else:
                res = []
        except:
            res = []
        return res

    def generalize4(self, term):
        res = []
        try:
            cats = wn.synsets(term)
            res = cats
        except:
            pass
        return res

    def generalize_term(self, term):
        res = {}
        res['gen1'] = generalize(term)
        res['gen2'] = generalize2(term, res['gen1'])['generalizations2']
        res['gen3'] = generalize3(term)
        res['gen4'] = generalize4(term)
        return res

# Bert encoder
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, TFBertModel, BertTokenizerFast, BatchEncoding
from tokenizers import Encoding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MyBertEmbedding(object):
    
    def __init__(self, size='base', merge_type='mean'):
        self.size = size
        self.merge_type = merge_type
        MODEL_NAME = 'bert-%s-cased' % self.size
        if 'large' in MODEL_NAME:
            self.vector_size = 1024
        else:
            self.vector_size = 768
        # MODEL_NAME = 'google/bert_uncased_L-12_H-768_A-12'
        self.config = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
        self.model = TFBertModel.from_pretrained(MODEL_NAME, config=self.config)
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        self.tokens = []
        self.vecs = []
        self.visited = []
        self._get_not_visited_only = True
    
    def sentence_embedding(self, sent):
        # Calculating original tokens' offsets
        offsets = [(i + sum([len(s) for s in sent[:i]]), tok) for i, tok in enumerate(sent)]
        # Doing the tokenization using bert
        r = self.tokenizer(' '.join(sent), padding=True, truncation=True, max_length=512, return_tensors='tf')
        output = self.model(r)
        # Calculating the representation using the last 4 layers from bert encoder
        res = np.zeros((len(r[0].ids), self.vector_size))
        for i, layer_output in enumerate(output[2]):
            if i >= 9:
                res += layer_output.numpy()[0]
        # Grouping the representation from the sub-tokens
        j = 0
        rr = {(0, sent[0]): []}
        for (start, end), vec in zip(r[0].offsets[1:-1], res[1:-1]):
            if start > offsets[j][0] + len(offsets[j][1]):
                j += 1
                rr[(j, sent[j])] = []
            rr[(j, sent[j])].append((' '.join(sent)[start:end], vec))
        # Merging the representation of the sub-tokens
        self.tokens = []
        self.sub_tokens = []
        self.vecs = []
        self.visited = []
        for i, t in rr:
            self.tokens.append(t)
            self.sub_tokens.append([])
            self.vecs.append([])
            for sub_t, v in rr[(i, t)]:
                # if sub_t != '_':
                self.sub_tokens[-1].append(sub_t)
                self.vecs[-1].append(v)
            self.vecs[-1] = np.array(self.vecs[-1])
            self.visited.append(False)
    
    def similarity(self, e1, e2):

        try:
            self._get_not_visited_only = False
            vec1 = self[e1]
            self._get_not_visited_only = True
            vec2 = self[e2]
            # print(len(vec1), len(vec2))
        except:
            print('error')
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def __contains__(self, x):
        return False
    
    def __getitem__(self, ind):
        if not self._get_not_visited_only:
            for i, t in enumerate(self.tokens):
                if t == ind:
                    tt = self.sub_tokens[i]
                    _index = tt.index('_') if '_' in tt and tt.index('_') > 0 else len(tt)
                    return self.vecs[i][:_index].mean(0) if self.merge_type == 'first' else self.vecs[i].mean(0)
        else:
            for i, t in enumerate(self.tokens):
                if t == ind and not self.visited[i]:
                    self.visited[i] = True
                    tt = self.sub_tokens[i]
                    _index = tt.index('_') if '_' in tt and tt.index('_') > 0 else len(tt)
                    return self.vecs[i][:_index].mean(0) if self.merge_type == 'first' else self.vecs[i].mean(0)
        return -1
    
    def __setitem__(self, ind, val):
        pass


# Evaluate class
from itertools import combinations, chain
import copy
import matplotlib.pyplot as plt
import gensim
import pandas as pd
import ast

class Evaluator(object):
    def __init__(self, loader, emb):
        self.loader = loader
        self.emb = emb
        mdl = emb.model
        self.noun_phrase = {}
        if type(mdl) == gensim.models.word2vec.Word2Vec or type(mdl) == gensim.models.FastText:
            self.model = mdl.wv
        else:
            self.model = mdl
            
    def _get_best_threshold(self, key, documents):
        doc2 = [(e, l, p) for e, l, p in documents[key] if 'B' in l or 'I' in l]
        entities, labels, poss = zip(*doc2)
        actor = entities[0]
        sims = [self.similarity(actor, e, pos) for e, pos in zip(entities, poss)]
        #print(sims)
        m = 1
        for sim in sims:
            if m > sim and sim > -1:
                m = sim
        return m
    
    def get_best_thresholds(self, documents):
        return [self._get_best_threshold(key, documents) for key in documents]
    
    def is_chunk(self, pos, lbl=None, sim=None, threshold=None):
        ps = pos.split('_')
        flag = False
        for p in ps:
            if p.startswith('N') or p == 'FW' or p == 'CD' or p.startswith('J'):
                flag = True
                break
        return flag and ((sim >= threshold) if not (sim is None) else True) and (('B' in lbl or 'I' in lbl) if not lbl is None else True)
        #return '_' in pos or pos.startswith('N')
        
    def is_correct_chunk(self, pos, microlbl, sim, threshold):
        return self.is_chunk(pos, sim=sim, threshold=threshold) and ('B' in microlbl or 'I' in microlbl)
        
    def is_labeled_entity(self, microlbl):
        count = sum([1 for c in microlbl if c is 'B'])
        return 'B' in microlbl, count
    
    def is_correct_labeled_entity(self, microlbls, sims, j, threshold):
        correct = False
        if sims[j] >= threshold:
            correct, count = self.is_labeled_entity(microlbls[j])
        count = 1 if not correct else count
        j += 1 if not correct else 0
        if correct:
            j = j + 1
            if not microlbls[j - 1].endswith('O'):
                while correct and j < len(microlbls) and microlbls[j].startswith('I'):
                    correct = correct and sims[j] >= threshold
                    j = j + 1
        return count - 1 if not correct else count, j
        #return self.is_labeled_entity(microlbl) and sim >= threshold
    
    def similarity(self, e1, e2, pos, all_comps=False, tag='word2vec'):
        e1 = e1.lower()
        e2 = e2.lower()
        if tag not in ['word2vec', 'fasttext']:
            return 1.0 if '{' in e2 and '}' in e2 else -1.0
        #if '_' in e2 and e2 not in self.model.vocab:
        #    
        #if not e1 in self.model.vocab:
        #    return -2.0
        if all_comps:
            allsubsets = lambda n: list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))
            names = np.array(e1.split('_'))
            subsets = np.delete(allsubsets(len(names)), 0)
            e1 = []
            for s in subsets:
                st = '_'.join(list(names[list(s)]))
                if st in self.model.vocab:
                    e1.append(st)
                    
            sims = [self.model.similarity(e, e2) if self.is_chunk(pos) else -2.0 for e in e1]
            #print(sims, e1, e2, max(sims))
            #return sum(sims) / len(sims)
            return max(sims)
        else:
            try:
                return self.model.similarity(e1, e2) if pos is None or self.is_chunk(pos) else -2.0
            except:
                return -2.0
                
    def spl_dict(self, originaltext, text, tokens, sp=None):
        words = text.split(sp)
        words = [w for w in words if w != '']
        pos = 0
        chunks = []
        chunks2 = []
        for w in words:
            is_chunk = '[' in w and ']' in w and '{' in w and '}' in w
            if is_chunk:
                tt = w[w.find('[') + 1: w.find(']')]
                is_chunk = True
            else:
                tt = w
            ts = tt.split('_')
            if is_chunk:
                chunk_start = -1
                for t in ts:
                    pos = originaltext.find(t, pos)
                    if chunk_start == -1:
                        chunk_start = pos
                chunk_end = pos + len(t)
                chunks.append((w, chunk_start, chunk_end))
        res = self.loader.apply_chunk(originaltext, tokens, chunks, originalChunk=True)
        return res
                
            
        #d = OrderedDict()
        #pos = 0
        #for w in words:
        #    new_pos = text.find(w, pos)
        #    if w != '':
        #        d[new_pos] = w
        #    pos = new_pos
        #return d
    
    def evaluate(self, key, documents, soups, threshold=0.5, all_comps=False, tag='word2vec'):
        document = documents[key]
        entities, labels, poss = zip(*document)
        # only for bert embedding:
        if isinstance(self.model, MyBertEmbedding):
            self.model.sentence_embedding(entities)

        if tag != 'word2vec':
            chunks = self.spl_dict(soups[key].find('originalabstract').text, soups[key].find(tag).text,
                                   ast.literal_eval(soups[key].find('stanford_tokens').text[len('OrderedDict('):-1]))
            #chunks = [(w, w.count('_') + 1) for w in d]
            entities, labels, poss = zip(*self.loader.get_labels(soups[key], chunks))
        actor = entities[0]
        sims = [self.similarity(actor, e, pos, all_comps=all_comps, tag=tag) for e, pos in zip(entities, poss)]
        # print(entities)
        #print(labels)
        # print(sims)
        #print()
        #print(list(zip(entities, sims)))
            
        precision, recall, f1 = self.pr_re_f1(sims, key, documents, soups, threshold=threshold, tag=tag)
        #print(precision, recall, f1)
        #logging.info("precision: %4f, recall: %4f, f1: %4f, Actor: %s" % (precision, recall, f1, actor))
        return precision, recall, f1
    
    def pr_re_f1(self, sims, key, documents, soups, threshold=0.5, tag='word2vec'):
        document = documents[key]
        entities, labels, poss = zip(*document)
        if tag != 'word2vec':
            chunks = self.spl_dict(soups[key].find('originalabstract').text, soups[key].find(tag).text,
                                   ast.literal_eval(soups[key].find('stanford_tokens').text[len('OrderedDict('):-1]))
            #chunks = [(w, w.count('_') + 1) for w in d]
            entities, labels, poss = zip(*self.loader.get_labels(soups[key], chunks))
            #print(entities)
        #tp, tn, fp, fn = 0, 0, 0, 0
        #for i in range(len(entities)):
        #    if sims[i] >= threshold:
        #        if labels[i] != 'O':
        #            tp = tp + 1
        #        else:
        #            fp = fp + 1
        #    else:
        #        if labels[i] != 'O':
        #            fn = fn + 1
        #        else:
        #            tn = tn + 1
        #precision = tp / (tp + fp) 
        #recall = tp / (tp + fn)
        correct_chunks = 0.0
        chunks = 0.0
        for i in range(len(entities)):
            correct_chunks += 1 if self.is_correct_chunk(poss[i], labels[i], sims[i], threshold) else 0
            #if self.is_chunk(poss[i], sim=sims[i], threshold=threshold):
            #if self.is_chunk(poss[i], sim=sims[i], threshold=threshold) and \
            #    not self.is_correct_chunk(poss[i], labels[i], sims[i], threshold):
            #    print("entity: %s, microLabel: %s, sim: %s, correct_chunks: %s" 
            #          % (entities[i], labels[i], sims[i], correct_chunks))
            #chunks += 1 if self.is_chunk(poss[i], lbl=labels[i]) else 0
            chunks += 1 if self.is_chunk(poss[i], sim=sims[i], threshold=threshold) else 0
            if self.is_chunk(poss[i], sim=sims[i], threshold=threshold):
                if not entities[i] in self.noun_phrase:
                    self.noun_phrase[entities[i]] = 0
                self.noun_phrase[entities[i]] += 1
        #print("Key: %s, Threshold: %s, #correct_chunks: %s, #chuncks: %s" % (key, threshold, correct_chunks, chunks))
        precision = correct_chunks / chunks if chunks > 0 else 0
        detected_entities = 0.0
        labeled_entities = 0.0
        for i in range(len(entities)):
            correct, count = self.is_labeled_entity(labels[i])
            labeled_entities += count if correct else 0
            #if correct:
            #    print(entities[i], labels[i], detected_entities, count)
        count = 0.0
        j = 0
        k = 0
        while j < len(labels):
            count, j = self.is_correct_labeled_entity(labels, sims, j, threshold)
            detected_entities += count
            #print("entity: %s, microLabel: %s, sim: %s, detected: %s, count: %s" 
            #      % (entities[k], labels[k], sims[k], detected_entities, count))
            k = j
        #print("Key: %s, Threshold: %s, #detected_entities: %s, #labeled_entities: %s" 
        #      % (key, threshold, detected_entities, labeled_entities))
        recall = detected_entities / labeled_entities if labeled_entities > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    def calc_mean(self, vec, documents, weighted_mean=False):
        wcount = sum([len(documents[key]) for key in documents])
        ws = [len(documents[key]) / wcount for key in documents]
        sum_ws = sum(ws)
        if weighted_mean:
            return np.dot(vec, ws) / sum_ws
        else:
            return sum(vec) / len(vec)
    
    def evaluate_all(self, documents, soups, threshold=0.5, all_comps=False, weighted_mean=False, tag='word2vec', silent=False):
        res = []
        if not silent:
            for key in tqdm(documents):
                res.append(self.evaluate(key, documents, soups, threshold=threshold, all_comps=all_comps, tag=tag))
        else:
            for key in documents:
                res.append(self.evaluate(key, documents, soups, threshold=threshold, all_comps=all_comps, tag=tag))
        precision, recall, f1 = zip(*res)
        p = self.calc_mean(precision, documents, weighted_mean=weighted_mean)
        r = self.calc_mean(recall, documents, weighted_mean=weighted_mean)
        f = self.calc_mean(f1, documents, weighted_mean=weighted_mean)
        #logging.info("Actor: {0}, precision: {1}, recall: {2}, f1: {3}".format("all actors", p, r, f))
        return p, r, f
    
    def coefficient_of_variation(self, documents, soups, threshold=0.5, all_comps=False, weighted_mean=False, tag='word2vec'):
        res = [self.evaluate(key, documents, soups, threshold=threshold, all_comps=all_comps, tag=tag) for key in documents]
        precision, recall, f1 = zip(*res)
        p = self.calc_mean(precision, documents, weighted_mean=weighted_mean)
        r = self.calc_mean(recall, documents, weighted_mean=weighted_mean)
        f = self.calc_mean(f1, documents, weighted_mean=weighted_mean)
        vp = np.var(precision)
        vr = np.var(recall)
        vf = np.var(f1)
        #logging.info("Actor: {0}, precision: {1}, recall: {2}, f1: {3}".format("all actors", p, r, f))
        return vp / p, vr / r, vf / f
    
    def add_ner(self, soup, file, ner_path='../../word2vec/wiki/abstract_xml/'):
        # soup2 = self.loader.read_soup(ner_path, file)
        soup2 = soup
        ner_tags = ['nertext7', 'nertext4', 'nertext3', 'presidio']
        for ner_tag in ner_tags:
            #tag = soup.new_tag(ner_tag)
            #tag.string = soup2.find(ner_tag).text
            #soup.doc.insert(-2, tag)
            soup.find(ner_tag).string = soup2.find(ner_tag).text
        return soup
    
    def anonymize(self, document, original_soup, file_name, threshold=0.5, path='evals/', all_comps=False):
        soup = copy.copy(original_soup)
        #print(poss)
        entities, labels, poss = zip(*document)
        actor = entities[0]
        sims = [self.similarity(actor, e, pos, all_comps=all_comps) for e, pos in zip(entities, poss)]
        t1 = ""
        t2 = ""
        for i in range(len(entities)):
            if sims[i] >= threshold:
                t1 += "{SENSITIVE[%s]} " % entities[i]
                t2 += "{SENSITIVE} "
            else:
                #if labels[i] == 'O':
                #    t1 += entities[i] + " "
                #else:
                #    t1 += "{%s[%s]} " % ("FALSE", entities[i])
                t1 += entities[i] + " "
                t2 += entities[i] + " "
        #tag = soup.new_tag('word2vec')
        #tag.string = t1
        #soup.doc.insert(-2, tag)
        soup.find('word2vec').string = t1
        soup = self.add_ner(soup, file_name)
        return soup
    
    def export(self, documents, soups, threshold=0.5, path='evals/', all_comps=False):
        newSoups = {}
        for key in tqdm(documents):
            soup = self.anonymize(documents[key], soups[key], key, threshold=threshold)
            if not os.path.isdir('./evals'):
                os.mkdir('./evals')
            newSoups[key] = self.loader.write_soup(path, key, soup)
        return newSoups
    
    def get_anon(self, soup, tag='word2vec'):
        #print(soup)
        tokens = soup.find(tag).text.split(None)
        tokens2 = []
        for t in tokens:
            if '[' in t and ']' in t and '{' in t and '}' in t:
                tt = (t[t.index('[')+1:t.index(']')], t[t.index('{')+1:t.index('[')])
            else:
                tt = (t, None)
            tokens2.append(tt)
        return tokens2
    
    def get_gen(self, path='../data/hits/', name='entities_generalizations4.csv', lower=True):
        df = pd.read_csv(path + name)
        gen = {}
        for i, row in df.iterrows():
            term = row['term'].lower() if lower else row['term']
            if not row['file_name'] in gen:
                gen[row['file_name']] = {}
            if not row['term'] in gen[row['file_name']]:
                gen[row['file_name']][term] = dict([('gen1', []), ('gen2', []), ('gen3', []), ('gen4', [])])
                gen[row['file_name']][term]['class'] = row['class']
            if row['generalizations'] != 'NONE':
                gen[row['file_name']][term]['gen1'] = ast.literal_eval(row['generalizations'])
            else:
                gen[row['file_name']][term]['gen1'] = []
            if row['generalizations2'] != 'NONE':
                gen[row['file_name']][term]['gen2'] = [t[(t.find('_') + 1 if '_' in t else 1):-1] for t in ast.literal_eval(row['generalizations2'])]
            else:
                gen[row['file_name']][term]['gen2'] = []
            if row['generalizations3'] != 'NONE':
                gen[row['file_name']][term]['gen3'] = ast.literal_eval(row['generalizations3'])
            else:
                gen[row['file_name']][term]['gen3'] = []
            if row['generalizations4'] != 'NONE':
                t = str(row['generalizations4']).replace('Synset(', '').replace(')', '').replace("['", '["')\
                .replace("']", '"]').replace(",'", ',"').replace("',", '",').replace(", '", ', "')
                gen[row['file_name']][term]['gen4'] = [t[:t.find('.')] for t in ast.literal_eval(t)]
            else:
                gen[row['file_name']][term]['gen4'] = []
        return gen
    
    def filter(self, w, term):
        filt = ['external links'] 
        filt.append(term)
        for f in filt:
            if f.lower() in w.lower():
                return False
        return True
    
    def get_get_list(self, term, gen):
        res = []
        res.extend(gen['gen2'])
        res.extend(gen['gen3'])
        res.extend(gen['gen4'])
        res.extend(gen['gen1'])
        return [w for w in res if self.filter(w, term)]
        
    def generalize(self, doc_id, document, original_soup, threshold=0.5, path='generalized/', tag='word2vec', all_comps=False, best_g=0):
        soup = copy.copy(original_soup)
        #print(poss)
        anon = self.get_anon(soup)
        gen = self.get_gen()
        entities, labels, poss = zip(*document)
        actor = entities[0]
        general = []
        for t in anon:
            new_t = None
            if not t[1] is None:
                try:
                    if not t[0] in gen[doc_id]:
                        gen[doc_id][t[0]] = generalize_term(t[0])
                    gen_list = self.get_get_list(t[0], gen[doc_id][t[0]])
                    sims = []
                    for w in gen_list:
                        sim = self.similarity(actor, w, None, all_comps=all_comps)
                        if sim < threshold:
                            sims.append((sim, w, t[0]))
                    if len(sims) > 0:
                        if best_g == 'max':
                            sim, w, tt = sorted(sims, key=lambda v: v[0])[-1]
                        elif best_g == 'min':
                            sim, w, tt = sorted(sims, key=lambda v: v[0])[0]
                        elif best_g == 'med':
                            sim, w, tt = sorted(sims, key=lambda v: v[0])[len(sims) // 2]
                        else:
                            sim, w, tt = sims[0]
                        new_t = '{%s[%s]}' % (w.replace(' ', '_'), tt)
                    else:
                        #new_t = '{%s[%s]}' % ((gen[doc_id][t[0]]['class'], t[0]) \
                        #                      if gen[doc_id][t[0]]['class'] != 'NONE' else ('SENSETIVE', t[0]))
                        new_t = '{%s[%s]}' % ((gen[doc_id][t[0]]['class'].replace(' ', '_'), t[0]) \
                                              if gen[doc_id][t[0]]['class'] != 'NONE' else ('WORLD', t[0]))
                except:
                    #new_t = '{%s[%s]}' % ('ERROR', t[0])
                    new_t = '{%s[%s]}' % ('WORLD', t[0])
            else:
                new_t = t[0]
            general.append(new_t)
        soup.find(tag + '_gen').string = ' '.join(general)
        #tag = soup.new_tag(tag + '_gen')
        #tag.string = ' '.join(general)
        #soup.doc.insert(-2, tag)
        return soup
            
    def export_generalized(self, documents, soups, threshold=0.5, path='generalized/', all_comps=False, best_g=0):
        newSoups = {}
        for key in tqdm(documents):
            soup = self.anonymize(documents[key], soups[key], key, threshold=threshold)
            soup2 = self.generalize(key, documents[key], soup, best_g=best_g, threshold=threshold)
            if not os.path.isdir('./generalized'):
                os.mkdir('./generalized')
            newSoups[key] = self.loader.write_soup(path, key, soup2)
        return newSoups
    
    def evaluate_plot(self, documents, soups, interval=0.05, all_comps=False, weighted_mean=False, tag='word2vec', 
                      plots={'P': True, 'R': True, 'F1': True}, filename=None):
        if filename is None:
            filename="figs/img_{0}_{1}_{2}_{3}.pdf".format(self.emb.vec_size, self.emb.window, self.emb.sg, self.emb.min_count)
        #title = "Title"
        xlabel = "threshold"
        ylabel = "Measures"
        metrics = ['precision', 'recall', 'f1', 'precision CV', 'recall CV', 'f1 CV']
        colors = ['b', 'g', 'r', 'c', 'y', 'w']
        markers = ['-o', '-^', '--', '-+', '-x', '-.']
        ts = np.arange(0.05, 1, interval)
        # ts = np.arange(-1, 1, interval)
        # ts = np.arange(-3, 1, interval)
        es = {'P' : [], 'R' : [], 'F1': [], 'PC': [], 'RC': [], 'FC': []}
        for t in tqdm(ts):
            p, r, f1 = self.evaluate_all(documents, soups, threshold=t, all_comps=all_comps, 
                                         weighted_mean=weighted_mean, tag=tag, silent=True)
            es['P'].append(p)
            es['R'].append(r)
            es['F1'].append(f1)
            if 'PC' in plots or 'RC' in plots or 'FC' in plots:
                pc, rc, fc = self.coefficient_of_variation(documents, soups, threshold=t, all_comps=all_comps, 
                                         weighted_mean=weighted_mean, tag=tag)
                es['PC'].append(pc)
                es['RC'].append(rc)
                es['FC'].append(fc)
        handlers = []
        for i in range(len(plots)):
            if isinstance(plots, dict):
                k = list(plots)[i] 
                if not plots[k]:
                    continue
            else:
                k = plots[i]
            handler, = plt.plot(ts, es[k], markers[i], label= metrics[i])
            handlers.append(handler)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
                
        #plt.title(title)
        plt.legend(handles=handlers, loc='best')
        if not os.path.isdir('./figs'):
            os.mkdir('./figs')
        plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return es


        