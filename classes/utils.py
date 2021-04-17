from bs4 import BeautifulSoup as BS
import ast

def get_original(text):
    temp = []
    for t in text.split():
        if t.startswith('{') and t.endswith('}'):
            temp.extend(t[t.index('[')+1:t.index(']')].split('_'))
        else:
            temp.append(t)
    return ' '.join(temp)

def load_presidio(soups):
    """Load presidio annotation from the file presidio.ann.txt"""
    presidio = {}
    with open('../data/presidio.ann.txt', 'r+', encoding='utf-8') as f:
        for line in f.read().split('\n\n\n'):
            head, tail = line.split('\n')
            presidio[head] = tail
    for key in presidio:     
        if key in soups:
            new_tag = soups[key].new_tag('presidio')
            x = 0
            ls = presidio[key].split()
            res = []
            for i, w1 in enumerate(ls):
                res.append((w1, []))
                for w2 in get_original(soups[key].nertext3.text).split()[x:]:
                    x += 1
                    if w2 != w1:
                        if w2 == ls[i+1]:
                            x -= 1
                            break
                        res[-1] = (w1, res[-1][1] + [w2])
                        # print(w2, end=' ')
                    else:
                        # print(w2)
                        break
            txt = []
            for w1, ws in res:
                if w1.startswith('<') and w1.endswith('>'):
                    txt.append('{%s[%s]}' % (w1[1:-1], '_'.join(ws)))
                else:
                    txt.append(w1)
            new_tag.string = ' '.join(txt)
            soups[key].doc.insert(-3, new_tag)
    return soups


from googleapiclient.discovery import build
import random
import time
import math
# import google api_id and api_key to get one you have to access:
# https://developers.google.com/maps/documentation/embed/get-api-key
try:
    from .secure import api_id, api_key
except:
    # print("secure.py not exist")
    api_id = "123" 
    api_key = "456"
with open("../data/hits/hits.txt", "rb") as f:
    hits = ast.literal_eval(f.read().decode("utf-8").strip())
oov = set()

def google_search(search_term, api_key, cse_id, **kwargs):
    """Search in google"""
    service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res

def results(words, hits=None):
    """Calculate the hits from google responses"""
    if hits is None:
        hits = {}
    if type(words) != list and type(words) != set:
        words = [words]
    rs = hits
    i = 0
    wait = 1.0
    for word in words:
        if word in hits.keys() and hits[word] != -1:
            i = i + 1
            continue
        try:
            time.sleep(wait)
            w = '%s' % word.replace('_', ' ')
            json_object = google_search(w, api_key, api_id)
            rs[word] = int(json_object['searchInformation']['totalResults'])
            # rs[word] = bing_search(w)
            # if i % 10 == 0:
            #     print("{0}, get the hits from bing for the entity {1} , count: {2}".format(i, word, rs[word]))
            i = i + 1
        except:
            rs[word] = -1
            #break
            pass
    return rs


def prop(s, hits):
    #print(s, hits[s])
    #print('a', hits['a'])
    if not s in hits or hits[s] < 0:
        oov.add(s)
        return avg / (hits['a'] + 1.0)
    return hits[s] / (hits['a'] + 1.0)

def IC(s, hits):
    s = s.lower()
    return -math.log(prop(s, hits), 2)

def my_split(t, spl=' ', glu='_'):
    entities = []
    tt = t.strip()
    index = tt.find('{')
    while index != -1:
        if len(tt[0:index].strip()) > 0:
            entities.extend(tt[0:index].strip().split(spl))
        ln = tt.find('}') + 1
        if len(tt[index:ln]) > 0:
            entities.append(tt[index:ln].strip().replace(' ', glu))
        tt = tt[ln:].strip()
        index = tt.find('{')
        
    entities.extend(tt.split(spl))
    return entities

def is_chunk(pos, lbl=None, sim=None, threshold=None):
        ps = pos.split('_')
        flag = False
        for p in ps:
            if p.startswith('N') or p == 'FW' or p == 'CD' or p.startswith('J'):
                flag = True
                break
        return flag and ((sim >= threshold) if not (sim is None) else True) \
                    and (('B' in lbl or 'I' in lbl) if not lbl is None else True)

def get_entities(docs):
    rows_list = []
    for key in tqdm(docs):
        #logging.info("file: {0}".format(key))
        i = 0
        actor = ''
        for term, lbl, pos in docs[key]:
            #if term == 'her_subsequent_roles':
            #    print(term)
            if is_chunk(pos, lbl):
                if i == 0 or term.lower() in actor.lower():
                    rows_list.append([key, i, term, 'ACTOR'])
                    actor = term
                else:
                    rows_list.append([key, i, term, 'NONE'])
                i = i + 1
                
    df = pd.DataFrame(rows_list, columns=['file_name', 'term_index', 'term', 'class'])
    return df

def spl_dict(originaltext, text, tokens, lower=True, sp=' ', loader=None):
    words = my_split(text, sp)
    words = [w for w in words if w != '']
    pos = 0
    chunks = []
    chunks2 = []
    for w in words:
        is_chunk = '[' in w and ']' in w and '{' in w and '}' in w
        if is_chunk:
            tt = w[w.find('[') + 1: w.find(']')].lower() if lower else w[w.find('[') + 1: w.find(']')]
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
    #print(chunks, originaltext)
    
    res = loader.apply_chunk(originaltext, tokens, chunks, originalChunk=True)
    return res
    

def get_chunk_lbl(soup, tag='word2vec_gen', lower=True, loader=None):
    if tag != 'word2vec_gen':
        tokens = ast.literal_eval(soup.find('stanford_tokens').text[len('OrderedDict('):-1])
        abstractpos = ast.literal_eval(soup.find('stanford_pos').text)
        microlbls = ast.literal_eval(soup.find('microlabels1').text)
        text = soup.find('originalabstract').text.lower()
    else:
        tokens = ast.literal_eval(soup.find('spacy_tokens').text[len('OrderedDict('):-1])
        abstractpos = ast.literal_eval(soup.find('spacy_pos').text)
        microlbls = ast.literal_eval(soup.find('microlabels2').text)
    if lower:
        text = soup.find('originalabstract').text.lower()
    else:
        text = soup.find('originalabstract').text
        
    chunks = spl_dict(text, soup.find(tag).text.strip(), tokens, loader=loader)
    
    j = 0
    res = []
    for i in range(len(chunks)):
        chunk = chunks[i][0]
        o = chunk
        l = chunks[i][1]
        if '[' in chunk and ']' in chunk and '{' in chunk and '}' in chunk:
            g = chunk[1:chunk.index('[')]
            o = chunk[chunk.index('[')+1:chunk.index(']')]
            if lower:
                o = o.lower()
        #l = len(o.split('_'))
        res.append((o ,''.join(microlbls[j:j+l]), '_'.join(abstractpos[j:j+l])))
        j += l
    return res, [c for c, l in chunks]

def Semantics(soup, tag='word2vec_gen', evaluate='p', lower=True, loader=None):
    #toks = my_split(soup.find(tag).text.strip())
    D, toks = get_chunk_lbl(soup, tag=tag, lower=lower, loader=loader)
    #print(D)
    #if soup.find('title').text.lower() == 'ben stiller' and tag == 'nertext7':
    #    del toks[0]
    ic = 0
    gs = []
    c = 0
    for i in range(len(D)):
        term, lbl, pos = D[i]
        g = None
        if is_chunk(pos, lbl):
            if evaluate == 'p':
                g = term
            elif evaluate == 'g':
                if '[' in toks[i] and ']' in toks[i] and '{' in toks[i] and '}' in toks[i]:
                    g = toks[i][1:toks[i].index('[')]
                    o = toks[i][toks[i].index('[')+1:toks[i].index(']')]
                    #print(c, toks[i], g, o)
                    if not g.isupper():
                        g = o + ' & ' + g
                    c += 1
                else:
                    g = term
            else:
                if '[' in toks[i] and ']' in toks[i] and '{' in toks[i] and '}' in toks[i]:
                    pass
                    c += 1
                else:
                    g = term
        else:
            if '[' in toks[i] and ']' in toks[i] and '{' in toks[i] and '}' in toks[i]:
                g = toks[i][1:toks[i].index('[')]
                o = toks[i][toks[i].index('[')+1:toks[i].index(']')]
                #print(c, toks[i], g, o)
                if not g.isupper():
                    g = o + ' & ' + g
                c += 1
        #print(term, g)
        if not g is None:
            if lower:
                term = term.lower()
                g = g.lower()
            results([term, g], hits)
            gs.append(g)
            ic += IC(g, hits)
    count = sum([1 if '[' in t and ']' in t and '{' in t and '}' in t else 0 for t in my_split(soup.find(tag).text.strip())])
    return ic, count
    
def Utility_preservation(soup, tag='word2vec_gen', evaluate='g', loader=None):
    s1, c1 = Semantics(soup, tag=tag, evaluate=evaluate, loader=loader)
    s2, c2 = Semantics(soup, tag=tag, evaluate='p', loader=loader)
    return s1 / s2 * 100, c1 + c2

def save_hits():
    """Save hits (in case there are newly emerged hits during the generalizing process)."""
    out_file = '../data/hits/hits.txt'
    with open(out_file, 'w+', encoding='utf-8') as outfile:
        outfile.write(str(hits))


from collections import OrderedDict
import pandas as pd
import wikipediaapi
from copy import deepcopy
wikipedia = wikipediaapi.Wikipedia('en')
from tqdm.auto import tqdm, trange
import spacy
nlp1 = spacy.load('en_core_web_md')

def load_original_article_from_wikipedia(soups):
    th = 300
    pages = OrderedDict()
    data = []
    max_count = 100
    count = 0
    gen_soups = OrderedDict()

    for key in tqdm(soups):
        if (soups[key].annotated.text == '1' and count <= max_count): # or\
    #     (soups[key].annotated.text == '0' and int(key[len('actor_'):-4]) % th == 0):
            count += 1 if soups[key].annotated.text == '1' else 0
            # if count >= 41:
            #     soups[key] = deepcopy(soups[key])
            #     soups[key].annotated.string = '0'
            # print(soups[key].title.text, soups[key].annotated.text, int(key[len('actor_'):-4]))
            # t = soups[key].doc.text[soups[key].doc.text.index('https://'):]
            # print(wikipedia.page(t[:t.find('\n')]).summary)
            pages[soups[key].title.text] = []
            for s in wikipedia.page(soups[key].title.text.replace(' ', '_')).sections: 
                if len(s.text) > 0:
                    # pages[soups[key].title.text].append(s.text)
                    # label = soups[key].title.text if soups[key].annotated.text == '1' else 'OTHER'
                    # data.append([key, s.text, label])
                    sents = [str(sent) for sent in nlp1(s.text).sents]
                    pages[soups[key].title.text].extend(sents)
                    label = soups[key].title.text if soups[key].annotated.text == '1' else 'OTHER'
                    for sent in sents:
                        data.append([key, sent, label])
            gen_soups[key] = soups[key]
    cols = ['key', 'text', 'label']
    original_pages_df = pd.DataFrame(data, columns=cols)
    return original_pages_df


def pre_processing(train_df, dev_df):
    unique_labels = dict(train_df['label'].value_counts())
    # for x in unique_labels:
    #     textcat.add_label(x)

        

    train_texts = [sent for sent in train_df['text'].values]
    train_labels = [{'cats': {label1: label1 == label for label1 in unique_labels}} for label in train_df['label']]

    dev_texts = [sent for sent in dev_df['text'].values]
    dev_cat_labels = dev_df['label'].values
    dev_labels = [{'cats': {label1: label1 == label for label1 in unique_labels}} for label in dev_df['label']]
    dev_docs = [nlp1.tokenizer(text) for text in dev_texts]
    return train_texts, train_labels, dev_texts, dev_labels, unique_labels


def add_annotaion_tag(gen_soups):
    for key in tqdm(gen_soups):
        text = gen_soups[key].find('originalabstract').text
        new_tag = gen_soups[key].new_tag('annotation')
        offset = 0
        output = ""
        for i, tag in enumerate(gen_soups[key].find('tags').findChildren('tag')):
            start, end = int(tag['offset']), int(tag['offset']) + int(tag['length'])
            output += text[offset:start] + ' '
            output += '{SENSETIVE[%s]}' % text[start:end].replace(' ', '_') + ' ' 
            offset = end
            # print(tag['value'] == text[start:end], tag['value'], text[start:end])
        output += text[offset:]
        new_tag.string = output.replace('  ', ' ').strip()
        gen_soups[key].doc.insert(-3, new_tag)
    return gen_soups