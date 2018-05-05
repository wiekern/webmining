#!/usr/bin/python3
import requests, json, os, math, sys
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer




def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
        
def remove_stopwords(words):
    stop = set(stopwords.words('english'))
    return [ word for word in words if word not in stop and word.isalpha()]



debug = 1
base_url = 'https://en.wikipedia.org/w/api.php?'
action = 'action=query&'
title = 'titles=Web%20design&'
prop = 'prop=links&format=json&pllimit=max'

if len(sys.argv) == 2 and sys.argv[1] == 'y':
    results = requests.get(base_url + action + title + prop)
    contents = json.loads(results._content)
    links_res = contents['query']['pages']['34035']['links']
    links_res.append({'title':'Web design'})
    for link in links_res:
        title_in_page = link['title']
        title = 'titles=' + title_in_page + '&'
        #prop = 'format=json&prop=revisions&rvprop=content&rvsection=0'
        prop = 'format=json&prop=extracts&exintro=&explaintext='
        results = requests.get(base_url + action + title + prop)
        contents = json.loads(results._content)
        pages = contents['query']['pages']
        for key in pages:
            try:
                content = pages[key]['extract']
                if os.path.isfile('./docs/' + title_in_page + '.txt'):
                    pass
                else:
                    with open('./docs/' + title_in_page + '.txt', "w+") as file:
                        
                        tokens = remove_stopwords(word_tokenize(content.lower()))
                        stemmer = SnowballStemmer("english")
                        s = [stemmer.stem(w) for w in tokens]
                        file.write(u' '.join(s))
            except KeyError:
                pass
            except IOError:
                print('Writing to doc [' + title +'] failed.')
            
      
# tokenize with nltk, removing stopwords
distinct_terms = {}
doc_tokenized = []
number_docs = 0
doc_lens = []   # for statistical model
for fs in listdir_nohidden('./docs'):
    with open('./docs/' + fs, 'r') as f:
        read_data = f.read()
        if read_data:
            number_docs += 1
            s = word_tokenize(read_data)
            doc_lens.append(len(s))
            doc_tokenized.append(s)

#print('number of docs: ', number_docs)


def tf_df_maxf(docs):
    terms = []
    max_freqs = []
    term_id = 0
    for i in range(len(docs)):
        new_doc = True
        considered = True
        max_freq = 0
        terms.append({})
        for term in docs[i]:
            #term = term.lower()
            # if distinct_terms.get(term, 0) == 0:
            #     #print(term, term_id)
            #     term_id += 1
            #     distinct_terms[term] = term_id
            if not terms[i]:
                terms[i] = {term: [1, 1]}
            elif terms[i].get(term, 'NA') == 'NA':
                terms[i][term] = [1, 1]
            else:
                terms[i][term][0] += 1  # the times of term appears in i-document 
                if considered:  # term appears in which documents
                    terms[i][term][1] += 1
                    considered = False


            if max_freq < terms[i][term][0]:
                max_freq = terms[i][term][0]
        max_freqs.append(max_freq)
    return terms, max_freqs

tf_df_matrix, max_freqs = tf_df_maxf(doc_tokenized)
if debug:
    #print(tf_df_matrix)
    #print(max_freqs)
    pass

queries = [ word_tokenize('web development design') ]

tf_df_in_query, max_freqs_in_query = tf_df_maxf(queries)

#print(tf_df_in_query)

# constructing terms matrix (i, j)
# print(len(distinct_terms))
# terms_matrix = np.zeros((len(distinct_terms), number_docs))
# for i in range(number_docs):
#     doc = tf_df_matrix[i]
#     for t in doc:
#         t_index = distinct_terms[t]
#         #print(t_index, len(terms_matrix[0]))
#         terms_matrix[t_index][i] = doc[t][0]

#print(terms_matrix)
#tf_df_matrix 
#[
# {'word1': [f_11 df_1], 'word2': [f_21 df_2]}, -- doc 1
# {'word1': [f_12 df_2], 'word2': [f_22 df_2]}, -- doc 2
#]
weight_tfidf = []
distance_d_matrix = []
for doc_index in range(len(tf_df_matrix)):
    weight = []
    doc_terms = tf_df_matrix[doc_index]
    distance_d = 0
    for key in doc_terms:
        w = doc_terms[key][0]/max_freqs[doc_index] * math.log(number_docs/doc_terms[key][1], 2)
        weight.append(w)
        distance_d += w ** 2
    distance_d = math.sqrt(distance_d)
    distance_d_matrix.append(distance_d)

    weight_tfidf.append(weight)
#print(weight_tfidf)


weight_query = []
distance_q_matrix = []
nominator_matrix = []
for doc_index in range(len(tf_df_in_query)):
    weight = []
    doc_terms = tf_df_in_query[doc_index]
    distance_q = 0
    w_ij = 0
    w_iq = 0
    for key in doc_terms:
        w_iq = 0.5 + 0.5 * doc_terms[key][0]/max_freqs_in_query[doc_index] * math.log(number_docs/doc_terms[key][1], 2)
        weight.append(w_iq)
        distance_q += w_iq ** 2
        for i in range(len(tf_df_matrix)):
            nominator = 0
            if tf_df_matrix[i].get(key, 0) == 0:
                pass
            else:
                w_ij = tf_df_matrix[i][key][0]/max_freqs[i] * math.log(number_docs/tf_df_matrix[i][key][1], 2)
            nominator += w_ij * w_iq
            nominator_matrix.append(nominator)

    distance_q = math.sqrt(distance_q)
    distance_q_matrix.append(distance_q)
    #print(nominator_matrix)
    weight_query.append(weight)

#print(weight_query)

def simularity(doc_index, query_index):
    return nominator_matrix[doc_index]/(distance_q_matrix[query_index]*distance_d_matrix[doc_index])

for i in range(number_docs):
    if i % 10 == 0:
        print('\n')

    print("%0.6f" % simularity(i, 0), end=" ")

# Statistical Language Model
# Pr(d_j | q) = Pr(q | d_j) * Pr(d_j)/Pr(q)

def probability(doc_index, query_index):
    doc = tf_df_matrix[doc_index]
    query = tf_df_in_query[query_index]
    pr_ti_dj = 1.0
    for key in doc:
        if query.get(key, 0) == 0:
            pass
        else:
            pr_ti_dj *= math.pow(doc[key][0]/doc_lens[doc_index], query[key][0])

    return pr_ti_dj

for i in range(number_docs):
    if i % 10 == 0:
        print('\n')

    print("%0.6f" % probability(i, 0), end=" ")







    
    
