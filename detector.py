"""Advanced duplicate content detection using AI/NLP techniques."""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import re
import json
from dataclasses import dataclass
from collections import defaultdict
from difflib import SequenceMatcher
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import langdetect
from textstat import flesch_reading_ease
from config import Config

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
class DuplicateResult:
    """Result of duplicate detection."""
    url1: str
    url2: str
    similarity_score: float
    similarity_type: str
    confidence: float
    common_content: str
    differences: str
    is_duplicate: bool
    metadata: Dict[str, any]


class TextPreprocessor:
    """Advanced text preprocessing for NLP tasks."""
    
    def __init__(self, language: str = 'english'):
        self.language = language
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words(language))
        except Exception:
            self.stop_words = set()
            
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove URLs, emails, and special characters
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.lower()
        
    def tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text."""
        tokens = word_tokenize(text.lower())
        return [self.stemmer.stem(token) for token in tokens 
                if token.isalnum() and token not in self.stop_words]
                
    def get_ngrams(self, text: str, n: int = 3) -> set:
        """Generate n-grams from text."""
        tokens = self.tokenize_and_stem(text)
        return {' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}


class SemanticAnalyzer:
    """Semantic analysis using transformer models."""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            pass
            
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get sentence embeddings for texts."""
        return self.model.encode(texts, show_progress_bar=False)
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        embeddings = self.get_embeddings([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key phrases using spaCy."""
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        phrases = []
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Multi-word phrases
                phrases.append(chunk.text.lower())
                
        # Remove duplicates and sort by frequency
        phrase_freq = defaultdict(int)
        for phrase in phrases:
            phrase_freq[phrase] += 1
            
        return [phrase for phrase, _ in sorted(
            phrase_freq.items(), key=lambda x: x[1], reverse=True
        )[:top_k]]
        
    def calculate_bert_score(self, text1: str, text2: str) -> float:
        """Calculate BERTScore between two texts."""
        try:
            P, R, F1 = bert_score([text1], [text2], lang='en', verbose=False)
            return float(F1[0])
        except Exception:
            return 0.0


class DuplicateDetector:
    """Main duplicate detection engine."""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.semantic_analyzer = SemanticAnalyzer(config.EMBEDDING_MODEL)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.85,
            min_df=2
        )
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_jaccard_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculate Jaccard similarity using n-grams."""
        ngrams1 = self.preprocessor.get_ngrams(text1, n)
        ngrams2 = self.preprocessor.get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
        
    def calculate_levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate Levenshtein similarity ratio."""
        return fuzz.ratio(text1, text2) / 100.0
        
    def calculate_sequence_similarity(self, text1: str, text2: str) -> float:
        """Calculate sequence similarity using difflib."""
        return SequenceMatcher(None, text1, text2).ratio()
        
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity."""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception:
            return 0.0
            
    def calculate_rouge_similarity(self, text1: str, text2: str) -> float:
        """Calculate ROUGE score similarity."""
        scores = self.rouge_scorer.score(text1, text2)
        return scores['rouge1'].fmeasure
        
    def detect_language(self, text: str) -> str:
        """Detect text language."""
        try:
            return langdetect.detect(text)
        except Exception:
            return 'en'
            
    def calculate_readability_score(self, text: str) -> float:
        """Calculate Flesch reading ease score."""
        try:
            return flesch_reading_ease(text)
        except Exception:
            return 0.0
            
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Create overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s.split()) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s.split())
                    
                current_chunk = overlap_sentences
                current_length = overlap_length
                
            current_chunk.append(sentence)
            current_length += sentence_length
            
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    def find_common_content(self, text1: str, text2: str) -> str:
        """Find common content between two texts."""
        matcher = SequenceMatcher(None, text1, text2)
        common_parts = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                common_parts.append(text1[i1:i2])
                
        return ' '.join(common_parts)
        
    def find_differences(self, text1: str, text2: str) -> Dict[str, str]:
        """Find differences between two texts."""
        matcher = SequenceMatcher(None, text1, text2)
        differences = {'added': [], 'removed': []}
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                differences['added'].append(text2[j1:j2])
            elif tag == 'delete':
                differences['removed'].append(text1[i1:i2])
            elif tag == 'replace':
                differences['removed'].append(text1[i1:i2])
                differences['added'].append(text2[j1:j2])
                
        return {
            'added': ' '.join(differences['added']),
            'removed': ' '.join(differences['removed'])
        }
        
    def calculate_comprehensive_similarity(self, content1: str, content2: str) -> Dict[str, float]:
        """Calculate comprehensive similarity using multiple methods."""
        # Preprocess texts
        clean1 = self.preprocessor.clean_text(content1)
        clean2 = self.preprocessor.clean_text(content2)
        
        if not clean1 or not clean2:
            return {}
            
        similarities = {
            'jaccard_3gram': self.calculate_jaccard_similarity(clean1, clean2, 3),
            'jaccard_4gram': self.calculate_jaccard_similarity(clean1, clean2, 4),
            'levenshtein': self.calculate_levenshtein_similarity(clean1, clean2),
            'sequence': self.calculate_sequence_similarity(clean1, clean2),
            'tfidf': self.calculate_tfidf_similarity(clean1, clean2),
            'rouge': self.calculate_rouge_similarity(clean1, clean2),
            'semantic': self.semantic_analyzer.calculate_semantic_similarity(clean1, clean2),
            'bert_score': self.semantic_analyzer.calculate_bert_score(clean1, clean2)
        }
        
        return similarities
        
    def detect_duplicates(self, contents) -> List[DuplicateResult]:
        """Detect duplicates among scraped contents."""
        from scraper import ScrapedContent
        results = []
        
        # Create content pairs
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                content1 = contents[i]
                content2 = contents[j]
                
                # Skip if content is too short
                if (len(content1.content) < self.config.MIN_CONTENT_LENGTH or
                    len(content2.content) < self.config.MIN_CONTENT_LENGTH):
                    continue
                    
                # Calculate similarities
                similarities = self.calculate_comprehensive_similarity(
                    content1.content, content2.content
                )
                
                if not similarities:
                    continue
                    
                # Weighted similarity score
                weights = {
                    'semantic': 0.25,
                    'bert_score': 0.20,
                    'jaccard_3gram': 0.15,
                    'tfidf': 0.15,
                    'rouge': 0.10,
                    'levenshtein': 0.10,
                    'sequence': 0.05
                }
                
                weighted_score = sum(
                    similarities[method] * weights.get(method, 0)
                    for method in similarities
                )
                
                # Determine if duplicate based on threshold
                is_duplicate = weighted_score >= self.config.SEMANTIC_THRESHOLD
                
                # Find common content and differences
                common = self.find_common_content(content1.content, content2.content)
                differences = self.find_differences(content1.content, content2.content)
                
                # Calculate confidence
                confidence = min(1.0, weighted_score * 1.2)
                
                result = DuplicateResult(
                    url1=content1.url,
                    url2=content2.url,
                    similarity_score=weighted_score,
                    similarity_type='comprehensive',
                    confidence=confidence,
                    common_content=common[:500] + '...' if len(common) > 500 else common,
                    differences=json.dumps(differences),
                    is_duplicate=is_duplicate,
                    metadata={
                        'individual_scores': similarities,
                        'content1_length': len(content1.content),
                        'content2_length': len(content2.content),
                        'word_count_ratio': min(
                            len(content1.content.split()),
                            len(content2.content.split())
                        ) / max(
                            len(content1.content.split()),
                            len(content2.content.split())
                        )
                    }
                )
                
                results.append(result)
                
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
    def find_near_duplicates(self, content: str, candidates: List[str], threshold: float = 0.75) -> List[Tuple[str, float]]:
        """Find near-duplicates for a given content."""
        similarities = []
        
        for candidate in candidates:
            score = self.semantic_analyzer.calculate_semantic_similarity(content, candidate)
            if score >= threshold:
                similarities.append((candidate, score))
                
        return sorted(similarities, key=lambda x: x[1], reverse=True)
