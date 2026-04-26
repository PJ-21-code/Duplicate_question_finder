# Hybrid Duplicate Question Detection Guide

This document outlines the core technologies and techniques used in our Duplicate Question Detection pipeline, explaining their purpose and role within the system.

## 1. Core Evaluation Pipeline

The system uses a multi-stage approach to evaluate if two questions mean the same thing, ranging from fast lexical checks to deep semantic understanding.

### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Library/Tool**: `scikit-learn` (`cv.pkl`)
- **Purpose**: Used for Stage 1 of the pipeline to calculate the exact word/lexical overlap between questions. 
- **Role**: Helps flag identical questions immediately (score > 0.9) to save compute, or drops entirely unrelated questions (score < 0.2).

### Sentence Embeddings (Bi-Encoder)
- **Model**: `all-mpnet-base-v2` (via `sentence-transformers`)
- **Purpose**: Used in Stage 3 to transform questions into dense semantic vectors. 
- **Role**: Allows the system to understand that differently worded questions might mean the same thing. By measuring cosine similarity between the embeddings, it catches paraphrases that TF-IDF misses. (Proceeds if score >= 0.4).

### Cross-Encoder
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `sentence-transformers`)
- **Purpose**: Used in Stage 4 as the deep semantic validation engine.
- **Role**: Unlike sentence embeddings which process sentences in isolation, the cross-encoder processes both questions simultaneously. This makes it highly accurate at picking up fine-grained contextual differences. It is the mandatory, primary and final decision maker for complex matches.

## 2. Text Processing and NLP Tools

### spaCy (`en_core_web_sm`)
- **Purpose**: Named Entity Recognition (NER).
- **Role**: Extracts `PERSON` entities (and other nouns) from questions to verify if the subject matters align even if wording differs. Accompanied by an alias mapping (e.g. mapping "narendra modi" to "narendra damodardas modi") to deal with naming variations.

### NLTK (Natural Language Toolkit)
- **Purpose**: Stopword removal.
- **Role**: Cleans common English words (like "the", "a", "is") out of the text during token-level feature extraction so the model focuses on the core nouns and verbs of the interrogative.

### Distance Metrics & FuzzyWuzzy
- **Library/Tool**: `distance`, `fuzzywuzzy` (`fuzz`)
- **Purpose**: Heuristic distance tracking.
- **Role**: Computes fuzzy string matching (like QRatio, Token Sort Ratio) and absolute/average length differences to generate traditional ML features for query points.

### BeautifulSoup
- **Purpose**: Text sanitization.
- **Role**: Cleans out HTML tags and formatting left inside questions before feeding them to the machine learning models.

## 3. Custom Utilities

### Soft Abbreviation Mapping
- **Purpose**: Lightweight dictionary replacement.
- **Role**: Replaces common domain terms (e.g. "ai" -> "artificial intelligence", "ml" -> "machine learning") early in the pipeline so both lexical and semantic models have richer context text to assess. 

## 4. UI Framework

### Streamlit
- **Purpose**: Web User Interface application (`duplicate_finder.py`).
- **Role**: Exposes the Python backend via a fast, reactive front-end. It provides caching (`@st.cache_resource`) so heavy ML models are only loaded into memory once and displays diagnostic pipeline metrics securely in realtime.
