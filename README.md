# AI-Powered Duplicate Content Detector

An advanced, AI-driven duplicate content detection system that leverages state-of-the-art NLP techniques to identify duplicate and near-duplicate content across websites at scale.

## 🚀 Features

### Core Capabilities
- **AI/NLP-Powered Detection**: Uses transformer models (BERT, Sentence-BERT) for semantic understanding
- **Multi-Method Analysis**: Combines 7+ similarity detection methods for maximum accuracy
- **Web-Scale Processing**: Concurrent scraping with intelligent rate limiting
- **Streamlit Interface**: Beautiful, interactive web interface
- **CLI Support**: Command-line interface for batch processing
- **Sitemap Support**: Automatically process entire websites via sitemap.xml

### Detection Methods
1. **Semantic Similarity** (BERT-based)
2. **TF-IDF Cosine Similarity**
3. **Jaccard Similarity** (3-gram & 4-gram)
4. **Levenshtein Distance**
5. **Sequence Matching**
6. **ROUGE Scores**
7. **BERTScore**

### Advanced Features
- **Intelligent Content Extraction**: Uses newspaper3k for accurate content extraction
- **Caching System**: Redis-like caching for improved performance
- **Language Detection**: Automatic language identification
- **Readability Analysis**: Flesch reading ease scores
- **Export Capabilities**: CSV, JSON, and detailed reports
- **Configurable Thresholds**: Fine-tunable similarity thresholds

## 🛠️ Installation

### Quick Start
```bash
# Clone or download the project
cd duplicate_content_detector

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Install spaCy model
python -m spacy download en_core_web_sm
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Streamlit Web Interface (Recommended)
```bash
# Launch the web interface
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### 2. Command Line Interface
```bash
# Analyze specific URLs
python utils.py --urls https://example.com/page1 https://example.com/page2

# Analyze from file
python utils.py --file urls.txt --output ./results

# Analyze entire website via sitemap
python utils.py --sitemap https://example.com/sitemap.xml
```

### 3. Python API
```python
from utils import DuplicateAnalyzer
from config import Config

# Initialize analyzer
config = Config()
config.SEMANTIC_THRESHOLD = 0.8
analyzer = DuplicateAnalyzer(config)

# Analyze URLs
results = asyncio.run(analyzer.analyze_urls([
    "https://example.com/page1",
    "https://example.com/page2"
]))

# Export results
analyzer.export_results(results)
```

## 📊 Configuration

### Environment Variables
```bash
# Web scraping
export MAX_WORKERS=10
export REQUEST_TIMEOUT=30
export RATE_LIMIT_DELAY=1.0

# Similarity thresholds
export SEMANTIC_THRESHOLD=0.75
export COSINE_THRESHOLD=0.85

# Content processing
export MIN_CONTENT_LENGTH=100
export MAX_CONTENT_LENGTH=50000
```

### JSON Configuration
Create `config.json`:
```json
{
    "SEMANTIC_THRESHOLD": 0.8,
    "COSINE_THRESHOLD": 0.85,
    "MIN_CONTENT_LENGTH": 150,
    "MAX_WORKERS": 15,
    "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
}
```

## 🔍 How It Works

### 1. Content Scraping
- **Intelligent Extraction**: Uses newspaper3k for article extraction
- **Fallback Methods**: BeautifulSoup extraction for complex pages
- **Rate Limiting**: Respects robots.txt and implements delays
- **Caching**: Stores scraped content for faster re-analysis

### 2. Preprocessing
- **Text Cleaning**: Removes HTML, scripts, and styling
- **Language Detection**: Identifies content language
- **Tokenization**: Advanced NLP tokenization
- **Stopword Removal**: Configurable stopword lists

### 3. Similarity Analysis
- **Multi-Method Approach**: Combines 7+ similarity metrics
- **Weighted Scoring**: Intelligent weighting of different methods
- **Confidence Scoring**: Provides confidence levels for results
- **Threshold-Based**: Configurable thresholds for duplicate detection

### 4. Results & Reporting
- **Interactive Visualizations**: Similarity heatmaps and distributions
- **Detailed Reports**: Common content and differences
- **Export Options**: CSV, JSON, and human-readable reports

## 📈 Performance

### Scalability
- **Concurrent Processing**: Up to 20 simultaneous requests
- **Memory Efficient**: Streaming processing for large datasets
- **Caching**: Reduces redundant processing
- **Configurable Limits**: Adjustable based on system resources

### Accuracy
- **High Precision**: BERT-based semantic understanding
- **Low False Positives**: Multi-method validation
- **Configurable Thresholds**: Balance precision vs recall
- **Confidence Scoring**: Reliability indicators for each result

## 🎨 Streamlit Interface Features

### Dashboard
- **Real-time Analysis**: Live progress updates
- **Interactive Visualizations**: Clickable charts and graphs
- **Responsive Design**: Works on desktop and mobile
- **Dark Mode Support**: Automatic theme switching

### Analysis Tools
- **URL Input**: Manual URL entry or sitemap processing
- **Batch Processing**: Handle hundreds of URLs
- **Advanced Filters**: Filter by similarity scores
- **Export Options**: Download results in multiple formats

## 🔧 Troubleshooting

### Common Issues

#### 1. SpaCy Model Missing
```bash
python -m spacy download en_core_web_sm
```

#### 2. NLTK Data Missing
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

#### 3. Memory Issues
- Reduce `MAX_WORKERS` in config
- Increase `MIN_CONTENT_LENGTH` to filter short content
- Use smaller batch sizes

#### 4. Rate Limiting
- Increase `RATE_LIMIT_DELAY`
- Reduce `MAX_WORKERS`
- Check robots.txt compliance

## 📋 Examples

### Basic Analysis
```python
# Simple URL analysis
urls = [
    "https://blog.example.com/post-1",
    "https://blog.example.com/post-2",
    "https://blog.example.com/post-3"
]

results = asyncio.run(analyzer.analyze_urls(urls))
print(f"Found {results['summary']['duplicates_found']} duplicates")
```

### Advanced Configuration
```python
# Custom configuration
config = Config()
config.EMBEDDING_MODEL = "all-mpnet-base-v2"
config.SEMANTIC_THRESHOLD = 0.85
config.MAX_WORKERS = 5

analyzer = DuplicateAnalyzer(config)
```

### Sitemap Analysis
```python
# Analyze entire website
results = analyzer.analyze_sitemap("https://example.com/sitemap.xml")
print(f"Analyzed {results['summary']['total_urls']} URLs")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions
- **Documentation**: Check the docs/ directory for detailed guides

## 🔄 Updates

Stay updated with the latest features:
- Follow the repository
- Check release notes
- Monitor performance improvements
