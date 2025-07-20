"""Optimized duplicate detection with Google Cloud NLP and batch processing."""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Tuple
import concurrent.futures
import time
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import hashlib
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import langdetect
from config import Config

# Google Cloud NLP
try:
    from google.cloud import language_v1
    from google.cloud.language_v1 import Document
    from google.oauth2 import service_account
    GOOGLE_NLP_AVAILABLE = True
except ImportError:
    GOOGLE_NLP_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class FastDuplicateResult:
    """Fast duplicate detection result."""
    url1: str
    url2: str
    similarity_score: float
    similarity_type: str
    is_duplicate: bool
    processing_time: float

class FastTextProcessor:
    """Fast text processing for bulk operations."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Fast text cleaning."""
        # Remove URLs, emails, and special characters
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = ' '.join(text.split())
        return text.lower()
        
    def tokenize_and_stem(self, text: str) -> List[str]:
        """Fast tokenization and stemming."""
        tokens = word_tokenize(text.lower())
        return [self.stemmer.stem(token) for token in tokens 
                if token.isalnum() and token not in self.stop_words]
                
    def get_ngrams(self, text: str, n: int = 3) -> set:
        """Generate n-grams efficiently."""
        tokens = self.tokenize_and_stem(text)
        return {' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

class GoogleCloudNLPDetector:
    """Google Cloud NLP-based duplicate detection."""
    
    def __init__(self, credentials_path: Optional[str] = None):
        if not GOOGLE_NLP_AVAILABLE:
            raise ImportError("Google Cloud NLP not available. Install google-cloud-language")
            
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = language_v1.LanguageServiceClient(credentials=credentials)
        else:
            self.client = language_v1.LanguageServiceClient()
            
    def analyze_content(self, content: str) -> Dict[str, any]:
        """Analyze content using Google Cloud NLP."""
        document = Document(
            content=content,
            type_=Document.Type.PLAIN_TEXT
        )
        
        response = self.client.analyze_entities(document=document)
        
        # Extract key entities for comparison
        entities = [entity.name.lower() for entity in response.entities]
        
        # Get sentiment as additional feature
        sentiment_response = self.client.analyze_sentiment(document=document)
        sentiment_score = sentiment_response.document_sentiment.score
        
        return {
            'entities': entities,
            'sentiment_score': sentiment_score,
            'content_hash': hashlib.md5(content.encode()).hexdigest()
        }
        
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity using Google Cloud NLP features."""
        analysis1 = self.analyze_content(content1)
        analysis2 = self.analyze_content(content2)
        
        # Entity similarity
        entities1 = set(analysis1['entities'])
        entities2 = set(analysis2['entities'])
        
        if not entities1 or not entities2:
            return 0.0
            
        entity_similarity = len(entities1.intersection(entities2)) / len(entities1.union(entities2))
        
        # Sentiment similarity
        sentiment_diff = abs(analysis1['sentiment_score'] - analysis2['sentiment_score'])
        sentiment_similarity = 1 - (sentiment_diff / 2)  # Normalize to 0-1
        
        # Content hash similarity (exact match check)
        hash_similarity = 1.0 if analysis1['content_hash'] == analysis2['content_hash'] else 0.0
        
        # Weighted combination
        return (entity_similarity * 0.7 + sentiment_similarity * 0.2 + hash_similarity * 0.1)

class FastDuplicateDetector:
    """Optimized duplicate detection for bulk processing."""
    
    def __init__(self, config: Config, use_google_nlp: bool = False, credentials_path: Optional[str] = None):
        self.config = config
        self.processor = FastTextProcessor()
        self.use_google_nlp = use_google_nlp
        
        if use_google_nlp and GOOGLE_NLP_AVAILABLE:
            self.google_detector = GoogleCloudNLPDetector(credentials_path)
            
    def calculate_jaccard_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Fast Jaccard similarity."""
        ngrams1 = self.processor.get_ngrams(text1, n)
        ngrams2 = self.processor.get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
        
    def calculate_tfidf_similarity(self, texts: List[str]) -> np.ndarray:
        """Fast TF-IDF similarity matrix."""
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf_matrix)
        
    def calculate_levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Fast Levenshtein similarity."""
        return fuzz.ratio(text1, text2) / 100.0
        
    def batch_detect_duplicates(self, contents: List['ScrapedContent']) -> List[FastDuplicateResult]:
        """Fast batch duplicate detection."""
        start_time = time.time()
        
        # Extract texts
        texts = [content.content for content in contents]
        urls = [content.url for content in contents]
        
        # Skip if too few texts
        if len(texts) < 2:
            return []
            
        # Fast TF-IDF similarity matrix
        similarity_matrix = self.calculate_tfidf_similarity(texts)
        
        results = []
        
        # Process pairs efficiently
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Multiple similarity metrics
                jaccard_sim = self.calculate_jaccard_similarity(texts[i], texts[j])
                levenshtein_sim = self.calculate_levenshtein_similarity(texts[i], texts[j])
                tfidf_sim = similarity_matrix[i, j]
                
                # Weighted combination for speed
                combined_score = (tfidf_sim * 0.5 + jaccard_sim * 0.3 + levenshtein_sim * 0.2)
                
                # Google Cloud NLP if enabled
                if self.use_google_nlp and hasattr(self, 'google_detector'):
                    google_sim = self.google_detector.calculate_similarity(texts[i], texts[j])
                    combined_score = (combined_score * 0.7 + google_sim * 0.3)
                
                results.append(FastDuplicateResult(
                    url1=urls[i],
                    url2=urls[j],
                    similarity_score=combined_score,
                    similarity_type='combined_fast',
                    is_duplicate=combined_score >= self.config.SEMANTIC_THRESHOLD,
                    processing_time=time.time() - start_time
                ))
                
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
    def find_exact_duplicates(self, contents: List['ScrapedContent']) -> List[FastDuplicateResult]:
        """Find exact duplicates using hash comparison."""
        results = []
        content_map = defaultdict(list)
        
        # Group by content hash
        for content in contents:
            content_hash = hashlib.md5(content.content.encode()).hexdigest()
            content_map[content_hash].append(content.url)
            
        # Find duplicates
        for urls in content_map.values():
            if len(urls) > 1:
                for i in range(len(urls)):
                    for j in range(i + 1, len(urls)):
                        results.append(FastDuplicateResult(
                            url1=urls[i],
                            url2=urls[j],
                            similarity_score=1.0,
                            similarity_type='exact_match',
                            is_duplicate=True,
                            processing_time=0.0
                        ))
                        
        return results
        
    def process_in_batches(self, contents: List['ScrapedContent'], batch_size: int = 50) -> List[FastDuplicateResult]:
        """Process large datasets in batches for memory efficiency."""
        all_results = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            batch_results = self.batch_detect_duplicates(batch)
            all_results.extend(batch_results)
            
        return sorted(all_results, key=lambda x: x.similarity_score, reverse=True)
