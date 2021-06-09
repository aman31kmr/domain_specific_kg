
import os
import pandas as pd
import re
import csv


import spacy
from spacy.attrs import intify_attrs
nlp = spacy.load("en_core_web_sm")

import neuralcoref

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from keras import backend as K
#pip install docx
#from docx import Document

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.python.keras import backend as k
#from tensorflow.keras.models import Sequential, load_model
#from tensorflow.keras.layers import LSTM, Dense, RepeatVector, Masking, TimeDistributed
#from tensorflow.keras.utils import plot_model

#from tensorflow.python.framework import ops
#ops.reset_default_graph()

#from keras import backend as K

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
all_stop_words = ['many', 'us', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                  'today', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                  'september', 'october', 'november', 'december', 'today', 'old', 'new']
all_stop_words = sorted(list(set(all_stop_words + list(stopwords.words('english')))))

abspath = os.path.abspath('') ## String which contains absolute path to the script file
#print(abspath)
os.chdir(abspath)

### ==================================================================================================
# Tagger

def get_tags_spacy(nlp, text):
    doc = nlp(text)    
    entities_spacy = [] # Entities that Spacy NER found
    for ent in doc.ents:
        entities_spacy.append([ent.text, ent.start_char, ent.end_char, ent.label_])
    return entities_spacy

import re
from spacy_lookup import Entity
textin = open(r"entities_mf.txt",'r', encoding='utf8')
txtall = textin.read()

list_vocab = []
vocab = txtall.split('\n')

for line in vocab:
    list_vocab.append(line)

list_vocab.sort(reverse=True)
    
# def get_tags_spacy(nlp, text):
    # entity = Entity(keywords_list=list_vocab)
    # nlp.add_pipe(entity, last=True)
    # doc = nlp(text)    
    # entities_spacy = [] # Entities that Spacy NER found
    # for ent in doc.ents:
        # entities_spacy.append([ent.text, ent.start_char, ent.end_char, ent.label_])
    # return entities_spacy

def tag_all(nlp, text, entities_spacy):
    if ('neuralcoref' in nlp.pipe_names):
        nlp.pipeline.remove('neuralcoref')    
    neuralcoref.add_to_pipe(nlp) # Add neural coref to SpaCy's pipe    
    doc = nlp(text)
    return doc

def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result

def tag_chunks(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    #print("tag chunks\n")
    #print(list(doc.ents))
    #print('\n')
    #print(list(doc.noun_chunks))
    spans = filter_spans(spans)
    #print(list(spans))
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': 'ENTITY'}, string_store))

def tag_chunks_spans(doc, spans, ent_type):
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': ent_type}, string_store))

def clean(text):
    text = text.strip('[(),- :\'\"\n]\s*')
    text = text.replace('—', ' - ')
    text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)(\"\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.\/)(\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z]{3,}\.)([A-Z]+[a-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)([A-Za-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    
    text = re.sub('’', "'", text, flags=re.UNICODE)           # curly apostrophe
    text = re.sub('‘', "'", text, flags=re.UNICODE)           # curly apostrophe
    text = re.sub('“', ' "', text, flags=re.UNICODE)
    text = re.sub('”', ' "', text, flags=re.UNICODE)
    text = re.sub("\|", ", ", text, flags=re.UNICODE)
    text = text.replace('\t', ' ')
    text = re.sub('…', '.', text, flags=re.UNICODE)           # elipsis
    text = re.sub('â€¦', '.', text, flags=re.UNICODE)          
    text = re.sub('â€“', '-', text)           # long hyphen
    text = re.sub('\s+', ' ', text, flags=re.UNICODE).strip()
    text = re.sub(' – ', ' . ', text, flags=re.UNICODE).strip()

    return text

def tagger(text):  
    df_out = pd.DataFrame(columns=['Document#', 'Sentence#', 'Word#', 'Word', 'EntityType', 'EntityIOB', 'Lemma', 'POS', 'POSTag', 'Start', 'End', 'Dependency'])
    corefs = []
    text = clean(text)
    
    nlp = spacy.load("en_core_web_sm")
    entities_spacy = get_tags_spacy(nlp, text)
    #print(entities_spacy,'\n')
    #print("SPACY entities:\n", ([ent for ent in entities_spacy]), '\n\n')
    document = tag_all(nlp, text, entities_spacy)
    #for token in document:
    #    print([token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_])
    
    ### Coreferences
    if document._.has_coref:
        for cluster in document._.coref_clusters:
            main = cluster.main
            for m in cluster.mentions:                    
                if (str(m).strip() == str(main).strip()):
                    continue
                corefs.append([str(m), str(main)])
    #print(corefs)
    tag_chunks(document)
    # for token in document:
        # print([token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_])
    
    
    # chunk - somethin OF something
    spans_change = []
    for i in range(2, len(document)):
        w_left = document[i-2]
        w_middle = document[i-1]
        w_right = document[i]
        if w_left.dep_ == 'attr':
            continue
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY' and (w_middle.text == 'of'): # or w_middle.text == 'for'): #  or w_middle.text == 'with'
            spans_change.append(document[w_left.i : w_right.i + 1])
    #print(spans_change)
    tag_chunks_spans(document, spans_change, 'ENTITY')
    #for token in document:
    #    print([token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_])
        
    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: verb + adp; verb + part 
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'ADP' or w_right.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: adp + verb; part  + verb
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_right.pos_ == 'VERB' and (w_left.pos_ == 'ADP' or w_left.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')
    
    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # # chunk all between LRB- -RRB- (something between brackets)
    # start = 0
    # end = 0
    # spans_between_brackets = []
    # for i in range(0, len(document)):
        # if ('-LRB-' == document[i].tag_ or r"(" in document[i].text):
            # start = document[i].i
            # continue
        # if ('-RRB-' == document[i].tag_ or r')' in document[i].text):
            # end = document[i].i + 1
        # if (end > start and not start == 0):
            # span = document[start:end]
            # try:
                # assert (u"(" in span.text and u")" in span.text)
            # except:
                # pass
                # #print(span)
            # spans_between_brackets.append(span)
            # start = 0
            # end = 0
    # tag_chunks_spans(document, spans_between_brackets, 'ENTITY')
            
    # chunk entities
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i-1]
        w_right = document[i]
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY':
            spans_change_verbs.append(document[w_left.i : w_right.i + 1])
    #print(spans_change_verbs)
    tag_chunks_spans(document, spans_change_verbs, 'ENTITY')
    
    doc_id = 1
    count_sentences = 0
    prev_dep = 'nsubj'
    for token in document:
        if (token.dep_ == 'ROOT'):
            if token.pos_ == 'VERB':
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]

            else:
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, prev_dep]

        else:
            df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_]
    
     #   print("||||||||||||||||||||||||||||||||||||||")
     #   print(token,df_out)
        if (token.text == '.'):
            count_sentences += 1
        prev_dep = token.dep_
        
    return df_out, corefs

### ==================================================================================================
### triple extractor

def get_predicate(s):
    pred_ids = {}
    for w, index, spo in s:
        if spo == 'predicate' and w != "'s" and w != "\"": #= 11.95
            pred_ids[index] = w
    predicates = {}
    for key, value in pred_ids.items():
        predicates[key] = value
    return predicates

def get_subjects(s, start, end, adps):
    subjects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'subject' in spo or 'entity' in spo or 'object' in spo:
                subjects[index] = w
    return subjects
    
def get_objects(s, start, end, adps):
    objects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'object' in spo or 'entity' in spo or 'subject' in spo:
                objects[index] = w
    return objects

def get_positions(s, start, end):
    adps = {}
    for w, index, spo in s:        
        if index >= start and index <= end:
            if 'of' == spo or 'at' == spo:
                adps[index] = w
    return adps

def create_triples(df_text, corefs):
    sentences = []
    aSentence = []
    
    for index, row in df_text.iterrows():
        d_id, s_id, word_id, word, ent, ent_iob, lemma, cg_pos, pos, start, end, dep = row.items()
        if 'subj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'subject'])
        elif 'ROOT' in dep[1] or 'VERB' in cg_pos[1] or pos[1] == 'IN':
            aSentence.append([word[1], word_id[1], 'predicate'])
        elif 'obj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'object'])
        elif ent[1] == 'ENTITY':
            aSentence.append([word[1], word_id[1], 'entity'])        
        elif word[1] == '.':
            sentences.append(aSentence)
            aSentence = []
        else:
            aSentence.append([word[1], word_id[1], pos[1]])
    
    relations = []
    #loose_entities = []
    for s in sentences:
        if len(s) == 0: continue
        preds = get_predicate(s) # Get all verbs
        """
        if preds == {}: 
            preds = {p[1]:p[0] for p in s if (p[2] == 'JJ' or p[2] == 'IN' or p[2] == 'CC' or
                     p[2] == 'RP' or p[2] == ':' or p[2] == 'predicate' or
                     p[2] =='-LRB-' or p[2] =='-RRB-') }
            if preds == {}:
                #print('\npred = 0', s)
                preds = {p[1]:p[0] for p in s if (p[2] == ',')}
                if preds == {}:
                    ents = [e[0] for e in s if e[2] == 'entity']
                    if (ents):
                        loose_entities = ents # not significant for now
                        #print("Loose entities = ", ents)
        """
        if preds:
            if (len(preds) == 1):
                #print("preds = ", preds)
                predicate = list(preds.values())[0]
                if (len(predicate) < 2):
                    predicate = 'is'
                #print(s)
                ents = [e[0] for e in s if e[2] == 'entity']
                #print('ents = ', ents)
                for i in range(1, len(ents)):
                    relations.append([ents[0], predicate, ents[i]])
                #print("relations :::: ", relations)
            pred_ids = list(preds.keys())
            pred_ids.append(s[0][1])
            pred_ids.append(s[len(s)-1][1])
            pred_ids.sort()
                    
            for i in range(1, len(pred_ids)-1):
                predicate = preds[pred_ids[i]]
                adps_subjs = get_positions(s, pred_ids[i-1], pred_ids[i])
                subjs = get_subjects(s, pred_ids[i-1], pred_ids[i], adps_subjs)
                adps_objs = get_positions(s, pred_ids[i], pred_ids[i+1])
                objs = get_objects(s, pred_ids[i], pred_ids[i+1], adps_objs)
                for k_s, subj in subjs.items():                
                    for k_o, obj in objs.items():
                        obj_prev_id = int(k_o) - 1
                        if obj_prev_id in adps_objs: # at, in, of
                            relations.append([subj, predicate + ' ' + adps_objs[obj_prev_id], obj])
                        else:
                            relations.append([subj, predicate, obj])
    
    ### Read coreferences: coreference files are TAB separated values
    coreferences = []
    for val in corefs:
        if val[0].strip() != val[1].strip():
            if len(val[0]) <= 50 and len(val[1]) <= 50:
                co_word = val[0]
                real_word = val[1].strip('[,- \'\n]*')
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
            else:
                co_word = val[0]
                real_word = ' '.join((val[1].strip('[,- \'\n]*')).split()[:7])
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
                
    # Resolve corefs
    triples_object_coref_resolved = []
    triples_all_coref_resolved = []
    for s, p, o in relations:
        coref_resolved = False
        for co in coreferences:
            if (s == co[0]):
                subj = co[1]
                triples_object_coref_resolved.append([subj, p, o])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_object_coref_resolved.append([s, p, o])

    for s, p, o in triples_object_coref_resolved:
        coref_resolved = False
        for co in coreferences:
            if (o == co[0]):
                obj = co[1]
                triples_all_coref_resolved.append([s, p, obj])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_all_coref_resolved.append([s, p, o])
    return(triples_all_coref_resolved)

### ==================================================================================================
## Get more using Network shortest_paths

def get_graph(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_edge(s, o, key=p, len=len(p)*50)
    return G

def get_entities_with_capitals(G):
    entities = []
    for node in G.nodes():
        if (any(ch.isupper() for ch in list(node))):
            entities.append(node)
    return entities

def get_paths_between_capitalised_entities(triples):
    
    g = get_graph(triples)
    ents_capitals = get_entities_with_capitals(g)
    paths = []
    #print('\nShortest paths among capitalised words -------------------')
    for i in range(0, len(ents_capitals)):
        n1 = ents_capitals[i]
        for j in range(1, len(ents_capitals)):
            try:
                n2 = ents_capitals[j]
                path = nx.shortest_path(g, source=n1, target=n2)
                if path and len(path) > 2:
                    paths.append(path)
                path = nx.shortest_path(g, source=n2, target=n1)
                if path and len(path) > 2:
                    paths.append(path)
            except Exception:
                continue
    return g, paths

def get_paths(doc_triples):
    triples = []
    g, paths = get_paths_between_capitalised_entities(doc_triples)
    f = plt.figure()
    nx.draw(g, nx.spring_layout(g))# ax=f.add_subplot(111))
    f.savefig("graph.png")

    for p in paths:
        path = [(u, g[u][v]['key'], v) for (u, v) in zip(p[0:], p[1:])]
        length = len(p)
        if (path[length-2][1] == 'in' or path[length-2][1] == 'at' or path[length-2][1] == 'on'):
            if [path[0][0], path[length-2][1], path[length-2][2]] not in triples:
                triples.append([path[0][0], path[length-2][1], path[length-2][2]])
        elif (' in' in path[length-2][1] or ' at' in path[length-2][1] or ' on' in path[length-2][1]):
            if [path[0][0], path[length-2][1], path[length-2][2]] not in triples:
                triples.append([path[0][0], 'in', path[length-2][2]])
    for t in doc_triples:
        if t not in triples:
            triples.append(t)
    return triples

def get_center(nodes):
    center = ''
    if (len(nodes) == 1):
        center = nodes[0]
    else:   
        # Capital letters and longer is preferred
        cap_ents = [e for e in nodes if any(x.isupper() for x in e)]
        if (cap_ents):
            center = max(cap_ents, key=len)
        else:
            center = max(nodes, key=len)
    return center



def draw_graph_centrality(G, dictionary):
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(G)
    #print("Nodes\n", G.nodes(True))
    #print("Edges\n", G.edges())


    nx.draw_networkx_nodes(G, pos, 
            nodelist=dictionary.keys(),
            with_labels=False,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size = [v * 150 for v in dictionary.values()],
            node_color='blue',
            alpha=0.5)
    edge_labels = {(u, v): d["p"] for u, v, d in G.edges(data=True)}
    #print(edge_labels)
    nx.draw_networkx_edge_labels(G, pos,
                           font_size=10,
                           edge_labels=edge_labels,
                           font_color='blue')
    nx.draw(G, pos, with_labels=True, node_size=1, node_color='blue')
        

    
def extract_triples(text):
    df_tagged, corefs = tagger(text)
    doc_triples = create_triples(df_tagged, corefs)
    all_triples = get_paths(doc_triples)
    # with open('tagger_result.txt', 'w') as f:
        # f.write('df_tagged'+'\n'+str(df_tagged)+'\n'+'\n')
        # f.write('corefs'+'\n'+str(corefs)+'\n'+'\n')
        # f.write('doc_triples'+'\n'+str(doc_triples)+'\n'+'\n')
        # f.write('all_triples'+'\n'+str(all_triples)+'\n'+'\n')
    
    filtered_triples = []    
    for s, p, o in all_triples:
        if ([s, p, o] not in filtered_triples):
            if s.lower() in all_stop_words or o.lower() in all_stop_words:
                continue
            elif s == p:
                continue
            if s.isdigit() or o.isdigit():
                continue
            if '%' in o or '%' in s: #= 11.96
                continue
            if (len(s) < 2) or (len(o) < 2):
                continue
            if (s.islower() and len(s) < 4) or (o.islower() and len(o) < 4):
                continue
            if s == o:
                continue            
            subj = s.strip('[,- :\'\"\n]*')
            pred = p.strip('[- :\'\"\n]*.')
            obj = o.strip('[,- :\'\"\n]*')
            
            for sw in ['a','an', 'the', 'its', 'their', 'his', 'her', 'our', 'all', 'old', 'new', 'latest', 'who', 'that', 'this', 'these', 'those']:
                subj = ' '.join(word for word in subj.split() if not word == sw)
                obj = ' '.join(word for word in obj.split()  if not word == sw)
            subj = re.sub("\s\s+", " ", subj)
            obj = re.sub("\s\s+", " ", obj)
            
            if subj and pred and obj:
                filtered_triples.append([subj, pred, obj])

    return filtered_triples

# def levenshtein_distance(str1, str2, ):
    # counter = {"+": 0, "-": 0}
    # distance = 0
    # for edit_code, *_ in ndiff(str1, str2):
        # if edit_code == " ":
            # distance += max(counter.values())
            # counter = {"+": 0, "-": 0}
        # else: 
            # counter[edit_code] += 1
    # distance += max(counter.values())
    # return distance
    
# def stemmer(word):
    # porter = PorterStemmer()
    # if word[-1]=='s':
       # stem_word = porter.stem(word)
    
    # return stem_word
        

# if stemmer(i[0]) or stemmer(i[0]):
    # stemmer(i
    

# def filtering_entities(mytriples):

    # triple_AI_ent = []
    # for i in mytriples:
        # if i[0] in list_vocab or i[2] in list_vocab :
            # triple_AI_ent.append(i)
    # return triple_AI_ent
def stemmer(word,list_vocab):
    ret=False
    porter = PorterStemmer()
    if word[-1]=='s':
       stem_word = porter.stem(word)
       if stem_word in list_vocab:
          ret= True
    return ret
            

def filtering_entities(mytriples):

    triple_AI_ent = []
    for i in mytriples:
            if (i[0] in list_vocab and i[2] in list_vocab) or (i[0] in list_vocab and stemmer(i[2],list_vocab) ) or (stemmer(i[0],list_vocab) and i[2] in list_vocab ) or (stemmer(i[0],list_vocab) and stemmer(i[2],list_vocab) ):
                #if (i[0] in list_vocab and i[2] in list_vocab) or (stemmer(i[0],list_vocab) and stemmer(i[2],list_vocab) ):
                triple_AI_ent.append(i)
    return triple_AI_ent

def context(triples,all_triples):
    result_triple=[]

    for i in range(0, len(all_triples)):
        if all_triples[i] in triples:
            sub = all_triples[i][0]
            #print("******************", sub)
            obj = all_triples[i][2]
            #print("******************", obj)
            #fwd
            index=i
            while not (sub[0]<='Z' and sub[0]>='A'):
                index-=1
                result_triple.append(all_triples[index])
                sub= all_triples[index][0]
            
            result_triple.append(all_triples[i])
            #bwd
            index=i 
            
            while obj==all_triples[index+1][0]:
                index+=1
                result_triple.append(all_triples[index])

    no_dupl_list = []
    for i in result_triple:
        if i not in no_dupl_list:
            no_dupl_list.append(i)
    
    return no_dupl_list

def fetch_sent(list_sent, text):
    sent = []
    if list_sent[1]  in text and list_sent[2] in text:
        sent.append(text)

    return sent

from nltk.stem import WordNetLemmatizer 

def lemma_final(triples):
  
    lemmatizer = WordNetLemmatizer() 
    new_triples=[]
    for triple in triples:
      sub= triple[0]
      obj= triple[2]
      new_sub= lemmatizer.lemmatize(sub.lower())
      new_obj= lemmatizer.lemmatize(obj.lower())
      new_triples.append([new_sub, triple[1] ,new_obj ])

    return new_triples

def one_word(triples, word):
  word_triples=[]
  for triple in triples:
    sub= triple[0]
    obj= triple[2]
    if sub==word or obj==word:
      word_triples.append(triple)
  return word_triples

    
if __name__ == "__main__":
    """
    If the environment is partially observable, however, then it could appear to be stochastic.
    """
    text = """Direct Energy Deposition (DED) systems are currently used to repair and maintain existing parts in the aerospace and automotive industries. Dense metals such as Steel and aluminium are used in additive manufacturing. 3D printing and additive manufacturing reflect that the technologies share the theme of material addition or joining throughout a 3D work envelope under automated control. Peter Zelinski, the editor-in-chief of Additive Manufacturing magazine, pointed out in 2017 that the terms are still often synonymous in casual usage,[7] but some manufacturing industry experts are trying to make a distinction whereby additive manufacturing comprises 3D printing plus other technologies or other aspects of a manufacturing process."""
    
    """
    Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years younger, with whom he shared an enthusiasm for computers. Paul and Bill used a teletype terminal at their high school, Lakeside, to develop their programming skills on several time-sharing computer systems.

    """
    """
    An arson fire caused an estimated $50,000 damage at a house on Mt. Soledad that was being renovated, authorities said Friday.San Diego police were looking for the arsonist, described as a Latino man who was wearing a red hat, blue shirt and brown pants, and may have driven away in a small, black four-door car.A resident on Palomino Court, off Soledad Mountain Road, called 9-1-1 about 9:45 a.m. to report the house next door on fire, with black smoke coming out of the roof, police said. Firefighters had the flames knocked down 20 minutes later, holding the damage to the attic and roof, said City spokesperson Alec Phillip. No one was injured.Metro Arson Strike Team investigators were called and they determined the blaze had been set intentionally, Phillip said.Police said one or more witnesses saw the suspect run south from the house and possibly leave in the black car.
    """

    #print("Line1")
    path = r"ch_mf.txt"
    def getText(filename):
        doc = open(filename, 'r')
        doc_read1 = doc.read()
        doc_read1= doc_read1.replace('- ','').replace('is illustrated in','').replace('Figure','')
        doc_read2=re.sub('[0-9].[0-9]+','', doc_read1)
        #print(doc_read2)
        doc_read = re.sub(r'\b[A-Z]+\b', '', doc_read2)
        return doc_read
    #print("Line2")    
    #text = getText(path)
    #print("Line3")
    text = text.lower()
    text_split = text.split('.')
    #print("Line4")
    mytriples = extract_triples(text)
    #print("Line5")
    mytriples_AI = filtering_entities(mytriples)
    #print("Line6")
    mytriples_AI = lemma_final(mytriples_AI)
    #mytriples_AI = one_word(mytriples_AI, 'agent')
    # list_sent_triple = []
    # for t in mytriples_AI:
        # #print(t) 
        # for i in text_split:
            # ans= fetch_sent(t,i)
            # if len(ans)>0:
                # print(i)
                # print(t)

    g, paths = get_paths_between_capitalised_entities(mytriples_AI)
    f = plt.figure(figsize=(15,10))
    #pos = nx.spring_layout(g)
    #nx.draw(g, pos, with_labels=True, node_size=10, node_color='blue')
    #nx.draw(g, nx.spring_layout(g))# ax=f.add_subplot(111))
    #nx.draw_planar(g,with_labels = True, alpha=0.8)
    #pos = nx.draw_planar(g,with_labels = True, alpha=0.8)
    pos=nx.random_layout(g)

    #plt.figure
    
    #print("Nodes\n", G.nodes(True))
    #print("Edges\n", G.edges())
    node_sizes = []
    for n in g.nodes:
        node_sizes.append( 100 * len(n)*2.5 )
    nx.draw_networkx_nodes(g, pos, 
            with_labels=False,
            edge_color='black',
            width=8,
            linewidths=1,
            node_size = node_sizes,
            node_color='blue',
            alpha=0.5)
    edge_labels = {(u, v): d['key'] for u, v, d in g.edges(data=True)}
    # #bbox = {'ec':[1,1,1,0], 'fc':[1,1,1,0]}
    # #print(edge_labels)
    nx.draw_networkx_edge_labels(g, pos, label_pos = 0.35,
                           font_size=8,
                           edge_labels=edge_labels,
                           font_color='blue')
    nx.draw(g, pos, with_labels=True, node_size=node_sizes, node_color='#00d5d9')
    f.savefig("graph.png")    
        
    for t in text_split:
        count=0
        for i in mytriples_AI: 
            #var=False
            if i[0]  in t and i[1]  in t and i[2] in t:
                if t.find(i[0]) < t.find(i[1]) and  t.find(i[1])< t.find(i[2]) :
            # if i[1] in t :
                # var= True
            # if i[2] in t and var:
                    #print(i)
                    count=1
        # if count>0:
            # print(t) 
    
    #mytriples_AI = context(mytriples_AI,mytriples) 
    #print('\n\nFINAL TRIPLES = ', len(mytriples_AI))
    #print("Line4")    
    for t in mytriples_AI:
       print(t)
