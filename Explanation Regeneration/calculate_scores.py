# -*- coding: utf-8 -*-
"""calculate_scores.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I-sBfNOF_h-xkufdHrmfosX88a8WJr2Y
"""

import os
import pandas as pd
import numpy as np
import logging
import json
import re
import string
import nltk
import random
from itertools import combinations
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

from google.colab import drive
drive.mount('/content/drive')
base_dir = 'drive/My Drive/knowledge engineering/assignments/assignment_4/'

class ProcessData:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def load_table(self, file_path):        
        with open(file_path) as f:
            table = f.readlines()   
        return table    

    def get_tables(self):
        file_list = os.listdir(self.input_directory)
        return file_list 

    def iterate_tables(self):
        processed_tables = []
        for file_name in self.get_tables():
            if file_name.endswith('.tsv'):
                file_path = os.path.join(self.input_directory, file_name)
                table = self.load_table(file_path)
                processed_table = self.process_text(table)
                processed_tables.append(processed_table)
        return processed_tables

    def create_sentence(self, row):       
        sentence = ' '.join(row)
        sentence += '.'
        return sentence
    
    def clean_row(self, row):
        row = row.lower()
        row = re.split('\t| ', row)[:-1]
        row = [i for i in row if len(i) != 0]
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        row = [re_punc.sub('', word) for word in row]        

        row = [word for word in row if word.isalpha()]       
        return row

    def process_text(self, table):
        processed_table = ''
        for row in table[1:]:
            processed_row = self.clean_row(row)
            processed_row = self.create_sentence(processed_row)
            processed_table += processed_row
        return processed_table

    def main(self):
        processed_tables = self.iterate_tables()
        return processed_tables

class CalculateScores:
    def __init__(self, processed_data, output_directory, pretrained=None):
        self.processed_data = processed_data
        self.output_directory = output_directory
        self.pretrained_embeddings = self.get_pretrained_embeddings(pretrained)
        self.pretrained_sample_terms = self.get_pretrained_sample_terms()
        self.pretrained_scores = {}

    def get_tfidf_scores(self):
        tfidf = TfidfVectorizer(ngram_range = (1,1))
        tfidf_matrix = tfidf.fit_transform(self.processed_data)
        matrix = pd.DataFrame(tfidf_matrix.toarray(), columns = tfidf.get_feature_names())
        tfidf_dict = matrix.max().to_dict()
        return tfidf_dict

    def prepare_word2vec(self):
        word2vec_data = []
        for table in self.processed_data:
            table = table.split('.')
            for sentence in table:
                word2vec_data.append(sentence.split(' '))
        return word2vec_data

    def calculate_word2vec(self):
        word2vec_data = self.prepare_word2vec()
        word_vectors = Word2Vec(word2vec_data, size=100, min_count=2, window=10)
        return word_vectors

    def upload_models(self, tfidf, word2vec):
        logging.info(f'Uploading models to {self.output_directory}')
        word2vec.wv.save(os.path.join(self.output_directory, 'word2vec.wordvectors'))
        with open(os.path.join(self.output_directory, 'tfidf.json'), 'w') as f:
            json.dump(tfidf, f)

    def get_pretrained_embeddings(self, pretrained):
        if pretrained is None:
            file_path = os.path.join(self.output_directory, 
                                    'GoogleNews-vectors-negative300.bin.gz')
            pretrained = KeyedVectors.load_word2vec_format(file_path, binary = True)
        return pretrained
    
    def get_pretrained_sample_terms(self):
        sample_terms = random.sample(list(self.pretrained_embeddings.wv.vocab), 25)
        return sample_terms

    def get_pretrained_embedding_score(self, token):        
        token_pretrained_score = 0
        for idx, sample_term in enumerate(self.pretrained_sample_terms):
            if token not in list(self.pretrained_embeddings.wv.vocab):
                return 0
            score = self.pretrained_embeddings.similarity(token, sample_term)
            token_pretrained_score += score
        token_pretrained_score /= len(self.pretrained_sample_terms)
        self.pretrained_scores[token] = token_pretrained_score
        return token_pretrained_score

    def rerank_tfidf(self):
        reranked_tfidf_scores = {}
        tfidf_scores = self.get_tfidf_scores()
        for idx, term in enumerate(list(tfidf_scores.keys())):
            if idx / 100 == 0:
                logging.info(f'reranked {idx} terms, on term {term}')
            score = tfidf_scores.get(term)
            pretrained_score = self.get_pretrained_embedding_score(term)
            score -= pretrained_score
            reranked_tfidf_scores[term] = score
        return reranked_tfidf_scores

    def rerank_tfidf_2(self, tfidf_matrix, word2vec):    
        embed_rerank = pd.DataFrame(0, columns = tfidf_matrix.columns, index=tfidf_matrix.index)    
        pre_embed_rerank = pd.DataFrame(0, columns = tfidf_matrix.columns, index=tfidf_matrix.index)    
        for doc_num in range(len(tfidf_matrix)):
            logging.info(f'Calculating rerank tfidf scores for document {doc_num}')
            filtered_terms = tfidf_matrix.iloc[doc_num][tfidf_matrix.iloc[doc_num] > 0]
            terms_combinations = combinations(filtered_terms.index, 1)
            filtered_terms = filtered_terms.sample(50)
            logging.info(f'Number of terms in document {doc_num} is {len(filtered_terms.index)}')
            sample_terms = random.sample(list(tfidf_matrix.iloc[doc_num][tfidf_matrix.iloc[doc_num] > 0].index), 20)
            for idx, term in enumerate(filtered_terms.index):
                logging.info(f'Term {term}')
                if idx / 100 == 0:
                    logging.info(f'Calculated scores for {idx} terms')
                for sample_term in sample_terms:
                    if (term not in list(self.pretrained_embeddings.wv.vocab)) or (sample_term not in list(self.pretrained_embeddings.wv.vocab)):
                        continue                
                    elif (term not in list(word2vec.wv.vocab)) or (sample_term not in list(word2vec.wv.vocab)):
                        continue
                    embed_rerank[term].iloc[doc_num] += word2vec.similarity(term, sample_term)
                    pre_embed_rerank[term].iloc[doc_num] += self.pretrained_embeddings.similarity(term, sample_term)
            embed_rerank.iloc[doc_num] /= len(sample_terms)
            pre_embed_rerank.iloc[doc_num] /= len(sample_terms)
            if doc_num == 2:
                break
        return embed_rerank, pre_embed_rerank

    def main(self):
        tfidf = self.rerank_tfidf()        
        word2vec = self.calculate_word2vec()
        self.upload_models(tfidf, word2vec)

def main(input_directory, output_directory, pretrained):
    processed_tables = ProcessData(input_directory, 
                                output_directory
                                ).main()    
        calculate_scores = CalculateScores(processed_tables, output_directory, pretrained)
    return calculate_scores

input_directory = os.path.join(base_dir, 'data/worldtree_full/tsv/tables')
output_directory = os.path.join(base_dir, 'data/mapping_files')
obj = main(input_directory, output_directory, pretrained).main()