#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import sys
import pandas as pd
import numpy as np
import random
import logging
import re
import nltk
import json
from statistics import mean 
from statistics import stdev
from gensim.models import KeyedVectors

from nltk.stem import PorterStemmer
from neo4j import GraphDatabase
nltk.download('stopwords')
from nltk.corpus import stopwords
logging.getLogger("gensim").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)


# In[16]:


class QueryDatabase:
    def __init__(self, terms):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("assn_4_data", "password"))
        self.terms = terms
        self.query = self.query()

    def close(self):
        self.driver.close()

    def query_database(self):
        with self.driver.session() as session:
            query = session.write_transaction(self._create_and_return_query)
        return query

    def _create_and_return_query(self, tx):
        result = tx.run(self.query)
        return result.values()
    
    def query(self):
        query = f"""
                MATCH (f:fact)-[term:SHARED_TERM]-(f:fact)
                WHERE [x IN {self.terms} WHERE x = term.shared_term]
                RETURN f.fact, f.fact_id
                """ 
        return query
    
    @staticmethod
    def separate_results(results):
        facts = [i[0] for i in results]
        fact_ids = [i[1] for i in results]
        return facts, fact_ids
    
    def main(self):
        logging.info(f'Getting facts for term {self.terms}')
        results = self.query_database()
        self.close()  
        facts, fact_ids = self.separate_results(results)
        return facts, fact_ids


# In[17]:


class AskQuestion:
    def __init__(self, question=None, directory=''):
        self.questions_table_path = 'tsv/questionsAndExplanations.tsv'
        self.questions_table = self.import_questions_table()
        self.question = question
        self.directory = directory
        self.initiate_questions = {}
        
    def import_questions_table(self):
        questions_table = pd.read_csv(self.questions_table_path, sep='\t', encoding='latin-1')
        return questions_table
        
    def get_question(self, sample):
        questions = self.questions_table['question'].tolist()[sample]
        return questions
    
    def get_explanation_ids(self, sample):
        explanations_raw = self.questions_table['explanation'].tolist()[sample]
        if explanations_raw is None:
            raise Exception('Looks like we do not have an answer to this question, lets find another')
        explanation_ids = []
        for explanation in explanations_raw.split(' '):
            explanation_ids.append(explanation.split('|')[0])
        return explanation_ids
    
    def get_random_sample(self):
        sample = random.sample(range(len(self.questions_table)), 1)[0]
        return sample
    
    def parse_question(self, question):
        question = question.split('?')[0].lower()
        if self.initiate_questions['random_question']:
            logging.info(f'Ok How about question: {question}')                
        question = question.split('(a)')[0]
        stop_words = set(stopwords.words('english'))
        tokens = question.split(' ')        
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [word for word in tokens if word.isalpha()]
        return tokens
    
    @staticmethod
    def thinking():
        responses = [
            'Hmm... let me think about that one',
            'That is a good question, the top results are',
            'You are really asking the tough ones huh',
            'What am I a machine, how am I supposed to know that? Lets see',
            'Good question',
            'Is that all you got?'
        ]
        response = random.sample(responses, 1)[0]
        logging.info(response)
        
    def initiate_question(self):
        logging.info('''Hello! How are you? I have mastered elementary school education. 
        Please ask me a question.''')
        question = input('If you have a question, please ask otherwise respond "no".').lower()
        if question.lower() == 'no':
            question = None
            logging.info('Okay let me find a good question for you.')
            self.initiate_questions['random_question'] = True
            score = input('Would you like to score it or just see responses?').lower()
            if ('score' in score) or ('yes' in score):
                self.initiate_questions['score'] = True
            else:
                self.initiate_questions['score'] = False
        else:
            self.initiate_questions['random_question'] = False
            self.initiate_questions['score'] = False
            return question
        questions_to_ask = ['score', 'random_question']
        for question_to_ask in questions_to_ask:
            if question_to_ask not in self.initiate_questions:
                logging.info('Lets start over because I missed a few details')
                self.initiate_question()
        self.thinking()
        
    def response(self, scores, explanation_ids=None):
        facts = [i[0] for i in scores][:10]
        if self.initiate_questions.get('random_question'):
            if self.initiate_questions.get('score'):
                self.score_question(scores, explanation_ids)
            else:
                logging.info('. '.join(facts))
        else:
            logging.info('. '.join(facts))
                
    def score_question(self, score, explanation_ids):
        # Getting only top 20 facts
        fact_ids = [i[1] for i in score][:20]
        explanation_id_count = 0
        for idx, explanation_id in enumerate(explanation_ids):            
            explanation_id_flag = False
            for fact_id in fact_ids:
                if explanation_id == fact_id:
                    logging.info(f'The explanation ID {explanation_id} was ranked at {idx + 1}')
                    explanation_id_flag = True
                    explanation_id_count += 1
            if explanation_id_flag == False:
                logging.info(f'The explanation ID {explanation_id} was not ranked at all')
        response = self.explanation_id_count_response(explanation_id_count, explanation_ids)
        logging.info(response)
                    
    @staticmethod
    def explanation_id_count_response(explanation_id_count, explanation_ids):
        metric = explanation_id_count / len(explanation_ids)
        if metric >= 0.5:
            response = f'{metric} is great'
        elif metric > 0.8:
            response = f'{metric}... wow. I think I am ready for middle school'
        elif metric >= 0.3:
            response = f'{metric} is not too bad'
        elif metric < 0.3:
            response = f'{metric} is really bad, I think I need to study more'
        return response
            
    def main(self):                
        question_response = self.initiate_question()
        if question_response is not None:
            explanation_ids = None
            question = question_response            
        else:
            sample = self.get_random_sample()            
            question = self.get_question(sample)
            explanation_ids = self.get_explanation_ids(sample)
        question_terms = self.parse_question(question)
        facts, fact_ids = QueryDatabase(question_terms).main()
        try:
            scores = CalculateScores(self.directory).calculate_scores(facts, question_terms,
                                                                     fact_ids)
            self.response(scores, explanation_ids)
            exit()
        except:
            raise Exception('Having trouble finding an answer, please ask another question')


# In[18]:


class CalculateScores:
    def __init__(self, directory=''):        
        self.directory = directory
    
    def load_tfidf(self):
        file_path = self.directory + 'mapping_files/tfidf.json'
        with open(file_path) as f:
            tfidf = json.load(f)
        return tfidf
    
    def load_word_vectors(self):
        file_path = self.directory + 'mapping_files/word2vec.wordvectors'
        word_vectors = KeyedVectors.load(file_path)
        return word_vectors
        
    def calculate_scores(self, facts, question_terms, fact_ids):
        tfidf = self.load_tfidf()
        word_vectors = self.load_word_vectors()
        tfidf_scores = []
        cosine_similarity_scores = []
        for fact in facts:
            tfidf_score = 0
            cosine_similarity_score = 0
            for token in fact.split(' '):
                tfidf_score_temp = tfidf.get(token)
                if tfidf_score_temp is None:
                    tfidf_score_temp = 0
                tfidf_score += tfidf_score_temp
                cosine_similarity_score += self.calculate_cosine_similarity(question_terms
                                                                    , token, word_vectors)
            tfidf_score /= len(fact)
            cosine_similarity_score /= len(fact)
            fact = ' '.join(fact)
            tfidf_scores.append(tfidf_score)
            cosine_similarity_scores.append(cosine_similarity_score)
        fact_scores = self.standardize_scores(facts, fact_ids, tfidf_scores, 
                                              cosine_similarity_scores)
        sorted_facts = sorted(fact_scores, key=lambda x: x[-1], reverse=True)
        return sorted_facts
    
    @staticmethod
    def standardize_scores(facts, fact_ids, tfidf_scores, cosine_similarity_scores):
        fact_scores = []
        tfidf_scores_mean = mean(tfidf_scores)
        tfidf_scores_stdev = stdev(tfidf_scores)
        cosine_similarity_scores_mean = mean(cosine_similarity_scores)
        cosine_similarity_scores_stdev = stdev(cosine_similarity_scores)
        for i in range(len(facts)):
            tfidf_zscore = (tfidf_scores[i] - tfidf_scores_mean) / tfidf_scores_stdev
            cosine_similarity_zscore = (cosine_similarity_scores[i] -                                         cosine_similarity_scores_mean) / cosine_similarity_scores_stdev
            agg_score = (tfidf_zscore + cosine_similarity_zscore) / 2
            fact_scores.append([facts[i], fact_ids[i], tfidf_zscore, 
                                cosine_similarity_zscore, agg_score])    
        return fact_scores
        
    @staticmethod
    def calculate_cosine_similarity(question_terms, token, word_vectors):
        cosine_similarity_score = 0
        for term in question_terms:
            if (term in word_vectors.vocab) and (token in word_vectors.vocab):                
                score = word_vectors.similarity(token, term)
                cosine_similarity_score += score
            else:
                cosine_similarity_score -= 1
                continue
        cosine_similarity_score /= len(question_terms)
        return cosine_similarity_score           


# In[22]:


if __name__ == '__main__':
    facts = AskQuestion().main()

