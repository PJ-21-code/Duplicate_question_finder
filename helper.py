import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import math
from spacy.cli import download
# --- CACHE MODELS FOR PERFORMANCE ---
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

nlp = load_spacy_model()
embedding_model = load_embedding_model()
cross_encoder = load_cross_encoder()
cv = pickle.load(open('cv.pkl','rb'))

def test_common_words(q1,q2):
  w1= set(map(lambda word: word.lower().strip(), q1.split(" ")))
  w2= set(map(lambda word: word.lower().strip(), q2.split(" ")))
  return len(w1&w2)

def test_word_total(q1,q2):
  w1= set(map(lambda word: word.lower().strip(), q1.split(" ")))
  w2= set(map(lambda word: word.lower().strip(), q2.split(" ")))
  return len(w1)+ len(w2)

def test_fetch_token_features(q1,q2):

  SAFE_DIV= 0.0001

  STOP_WORDS= stopwords.words('english')

  token_features= [0.0]*8

  q1_tokens= q1.split()
  q2_tokens= q2.split()

  if(len(q1_tokens)==0 or len(q2_tokens)==0):
    return token_features

  #getting words from questiions
  q1_words= set([word for word in q1_tokens if word not in STOP_WORDS])
  q2_words= set([word for word in q2_tokens if word not in STOP_WORDS])

  #Getting stopwords from questions
  q1_stop= set([word for word in q1_tokens if word in STOP_WORDS])
  q2_stop= set([word for word in q2_tokens if word in STOP_WORDS])

  #getting no. of common words
  common_words_count= len(q1_words.intersection(q2_words))

  #getting stopwords count
  common_stop_count= len(q1_stop.intersection(q2_stop))

  #count of common tokens
  common_tokens_count= len(set(q1_tokens).intersection(set(q2_tokens)))

  token_features[0]= common_words_count/(min(len(q1_words),len(q2_words))+ SAFE_DIV)
  token_features[1]= common_words_count/(max(len(q1_words),len(q2_words))+ SAFE_DIV)
  token_features[2]= common_stop_count/(min(len(q1_stop),len(q2_stop))+ SAFE_DIV)
  token_features[3]= common_stop_count/(max(len(q1_stop),len(q2_stop))+ SAFE_DIV)
  token_features[4]= common_tokens_count/(min(len(q1_tokens),len(q2_tokens))+ SAFE_DIV)
  token_features[5]= common_tokens_count/(min(len(q1_tokens),len(q2_tokens))+ SAFE_DIV)

  token_features[6]= int(q1_tokens[-1]==q2_tokens[-1])
  token_features[7]= int(q1_tokens[0]==q2_tokens[0])

  return token_features

def test_fetch_length_features(q1,q2):

  length_features= [0.0]*3

  q1_tokens= q1.split()
  q2_tokens= q2.split()

  if(len(q1_tokens)==0 or len(q2_tokens)==0):
     return length_features

  #absolute length
  length_features[0]= abs(len(q1_tokens)-len(q2_tokens))

  #average length
  length_features[1]= (len(q1_tokens)+len(q2_tokens))/2

  strs= list(distance.lcsubstrings(q1, q2))
  length_features[2]= len(strs[0])/ (min(len(q1),len(q2)) + 1)

  return length_features

def test_fetch_fuzzy_features(q1,q2):

  fuzzy_features= [0.0]*4

  fuzzy_features[0]= fuzz.QRatio(q1,q2)

  fuzzy_features[1]= fuzz.partial_ratio(q1,q2)

  fuzzy_features[2]= fuzz.token_sort_ratio(q1,q2)

  fuzzy_features[3]= fuzz.token_set_ratio(q1,q2)

  return fuzzy_features

# text preprocessing

def preprocess(q):

  q= str(q).lower().strip()

  q= q.replace('$','dollar')
  q= q.replace('%','percent')
  q= q.replace('₹','rupee')
  q= q.replace('€','euro')
  q= q.replace('@','at')

  # math appear around 900 times in dataset
  q= q.replace('[math]','')

  q= q.replace(',000,000,000','b')
  q= q.replace(',000,000','m')
  q= q.replace(',000','k')
  q= re.sub(r'([0-9]+)000000000',r'\1b',q)
  q= re.sub(r'([0-9]+)000000',r'\1m',q)
  q= re.sub(r'([0-9]+)000',r'\1k',q)

  # Source - https://stackoverflow.com/a
  # Posted by arturomp, modified by community. See post 'Timeline' for change history
  # Retrieved 2025-12-04, License - CC BY-SA 3.0

  contractions = {
   "ain't": "am not",
   "aren't": "are not",
   "can't": "cannot",
   "can't've": "cannot have",
   "'cause": "because",
   "could've": "could have",
   "couldn't": "could not",
   "couldn't've": "could not have",
   "didn't": "did not",
   "doesn't": "does not",
   "don't": "do not",
   "hadn't": "had not",
   "hadn't've": "had not have",
   "hasn't": "has not",
   "haven't": "have not",
   "he'd": "he had",
   "he'd've": "he would have",
   "he'll": "he shall",
   "he'll've": "he shall have",
   "he's": "he has",
   "how'd": "how did",
   "how'd'y": "how do you",
   "how'll": "how will",
   "how's": "how has",
   "I'd": "I had",
   "I'd've": "I would have",
   "I'll": "I shall",
   "I'll've": "I shall have",
   "I'm": "I am",
   "I've": "I have",
   "isn't": "is not",
   "it'd": "it had",
   "it'd've": "it would have",
   "it'll": "it shall",
   "it'll've": "it shall have",
   "it's": "it has",
   "let's": "let us",
   "ma'am": "madam",
   "mayn't": "may not",
   "might've": "might have",
   "mightn't": "might not",
   "mightn't've": "might not have",
   "must've": "must have",
   "mustn't": "must not",
   "mustn't've": "must not have",
   "needn't": "need not",
   "needn't've": "need not have",
   "o'clock": "of the clock",
   "oughtn't": "ought not",
   "oughtn't've": "ought not have",
   "shan't": "shall not",
   "sha'n't": "shall not",
   "shan't've": "shall not have",
   "she'd": "she had",
   "she'd've": "she would have",
   "she'll": "she shall",
   "she'll've": "she shall have",
   "she's": "she has",
   "should've": "should have",
   "shouldn't": "should not",
   "shouldn't've": "should not have",
   "so've": "so have",
   "so's": "so as",
   "that'd": "that would",
   "that'd've": "that would have",
   "that's": "that has",
   "there'd": "there had",
   "there'd've": "there would have",
   "there's": "there has",
   "they'd": "they had",
   "they'd've": "they would have",
   "they'll": "they shall",
   "they'll've": "they shall have",
   "they're": "they are",
   "they've": "they have",
   "to've": "to have",
   "wasn't": "was not",
   "we'd": "we had",
   "we'd've": "we would have",
   "we'll": "we will",
   "we'll've": "we will have",
   "we're": "we are",
   "we've": "we have",
   "weren't": "were not",
   "what'll": "what shall",
   "what'll've": "what shall have",
   "what're": "what are",
   "what's": "what has",
   "what've": "what have",
   "when's": "when has",
   "when've": "when have",
   "where'd": "where did",
   "where's": "where has",
   "where've": "where have",
   "who'll": "who shall",
   "who'll've": "who shall have",
   "who's": "who has",
   "who've": "who have",
   "why's": "why has",
   "why've": "why have",
   "will've": "will have",
   "won't": "will not",
   "won't've": "will not have",
   "would've": "would have",
   "wouldn't": "would not",
   "wouldn't've": "would not have",
   "y'all": "you all",
   "y'all'd": "you all would",
   "y'all'd've": "you all would have",
   "y'all're": "you all are",
   "y'all've": "you all have",
   "you'd": "you had",
   "you'd've": "you would have",
   "you'll": "you will",
   "you'll've": "you will have",
   "you're": "you are",
   "you've": "you have"

  }


  q_decontracted=[]
  for word in q.split():
     if word in contractions:
        word= contractions[word]

     q_decontracted.append(word)

  q= ' '.join(q_decontracted)
  q= q.replace("'ve"," have")
  q= q.replace("n't"," not")
  q= q.replace("'re"," are")
  q= q.replace("'ll"," will")

  q= BeautifulSoup(q)
  q= q.get_text()
  pattern= re.compile(r'\W')
  q= re.sub(pattern,' ',q).strip()

  return q



def query_point_creator(q1,q2):

  input_query=[]

  q1= preprocess(q1)
  q2= preprocess(q2)

  input_query.append(len(q1))
  input_query.append(len(q2))

  input_query.append(len(q1.split(' ')))
  input_query.append(len(q2.split(' ')))

  input_query.append(test_common_words(q1,q2))
  input_query.append(test_word_total(q1,q2))
  input_query.append(round(test_common_words(q1,q2)/test_word_total(q1,q2),2))

  token_features= test_fetch_token_features(q1,q2)
  input_query.extend(token_features)

  length_features= test_fetch_length_features(q1,q2)
  input_query.extend(length_features)

  fuzzy_features= test_fetch_fuzzy_features(q1,q2)
  input_query.extend(fuzzy_features)

  q1_bow= cv.transform([q1]).toarray()

  q2_bow= cv.transform([q2]).toarray()

  return np.hstack((np.array(input_query).reshape(1,-1),q1_bow,q2_bow))

# --- NEW HYBRID PIPELINE METHODS ---

def compute_lexical_similarity(q1, q2):
    # TF-IDF + cosine similarity
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()
    cos_sim = cosine_similarity(q1_bow, q2_bow)[0][0]
    
    # Token overlap / Jaccard similarity
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    if len(w1.union(w2)) == 0:
        jaccard = 0.0
    else:
        jaccard = len(w1.intersection(w2)) / len(w1.union(w2))
        
    return float(cos_sim), float(jaccard)

def extract_entities(q):
    doc = nlp(q)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

alias_map = {
    "modi": "narendra damodardas modi",
    "narendra modi": "narendra damodardas modi"
}

def normalize_entity(entity):
    entity = entity.lower().strip()
    return alias_map.get(entity, entity)

def compute_embedding_similarity(q1, q2):
    emb1 = embedding_model.encode([q1])
    emb2 = embedding_model.encode([q2])
    return float(cosine_similarity(emb1, emb2)[0][0])

def compute_cross_encoder_score(q1, q2):
    score = cross_encoder.predict([[q1, q2]])[0]
    # Apply sigmoid over logit output to convert it effectively into a Probability between 0-1
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    return sigmoid(float(score))

abbrev_map = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "ds": "data science",
    "cv": "computer vision"
}

def expand_abbreviations(text):
    words = text.split()
    expanded = [abbrev_map.get(w.lower(), w) for w in words]
    return " ".join(expanded)

def check_duplicate(q1, q2):
    metrics = {}
    
    # 1. Preprocess text (basic cleaning)
    q1_clean = preprocess(q1)
    q2_clean = preprocess(q2)
    
    # 1.5 Soft Abbreviation Handling
    q1_clean = expand_abbreviations(q1_clean)
    q2_clean = expand_abbreviations(q2_clean)
    
    # 2. Lexical Similarity (TF-IDF)
    cos_sim, jaccard = compute_lexical_similarity(q1_clean, q2_clean)
    metrics["TF-IDF Score"] = cos_sim
    metrics["Token/Jaccard Score"] = jaccard
    
    if cos_sim > 0.9:
        return "Duplicate", metrics
    if cos_sim < 0.2:
        return "Not Duplicate", metrics
        
    # Optional Entity Check to persist working logic, non-blocking
    ents1 = extract_entities(q1)
    ents2 = extract_entities(q2)
    
    norm_ents1 = set([normalize_entity(e) for e in ents1])
    norm_ents2 = set([normalize_entity(e) for e in ents2])
    
    entity_match_score = 0
    if len(norm_ents1) > 0 and len(norm_ents2) > 0 and len(norm_ents1.intersection(norm_ents2)) > 0:
        entity_match_score = 1
        
    metrics["Entity Match"] = "Yes" if entity_match_score == 1 else "No"
    
    # 3. Compute embedding similarity
    emb_sim = compute_embedding_similarity(q1_clean, q2_clean)
    metrics["Embedding Similarity"] = emb_sim
    
    # Fix Early Rejection Threshold
    if emb_sim < 0.4:
        return "Not Duplicate", metrics
        
    # 4. Deep Semantic Check (MANDATORY stage)
    ce_score = compute_cross_encoder_score(q1_clean, q2_clean)
    metrics["Cross-encoder Score"] = ce_score
    
    # 5. Final Decision Logic
    if ce_score > 0.75:
        return "Duplicate", metrics
    elif ce_score < 0.45:
        return "Not Duplicate", metrics
    else:
        return "Uncertain", metrics