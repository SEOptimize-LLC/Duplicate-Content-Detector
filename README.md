# AI-Powered Duplicate Content Detector

An advanced Streamlit application for detecting duplicate content across websites using state-of-the-art AI and NLP techniques. This tool combines multiple similarity detection methods including semantic analysis, TF-IDF, Jaccard similarity, and BERT embeddings to provide comprehensive duplicate detection.

## Features

### 🔍 **Advanced Detection Methods**
- **Semantic Analysis**: Uses transformer models (Sentence-BERT) for deep semantic understanding
- **TF-IDF Cosine Similarity**: Traditional but effective content similarity
- **Jaccard Similarity**: N-gram based similarity for surface-level duplicates
- **BERTScore**: State-of-the-art contextual similarity using BERT embeddings
- **ROUGE Scores**: For evaluating content overlap quality
- **Levenshtein Distance**: Character-level similarity for near-duplicates

### 🛠️ **Smart Web Scraping**
- **Intelligent Content Extraction**: Uses newspaper3k and BeautifulSoup for optimal content extraction
- **Rate Limiting**: Built-in delays to prevent 429 errors
- **Caching System**: Redis-like caching to avoid re-scraping
- **Sitemap Support**: Extract URLs directly from XML sitemaps
- **Error Handling**: Comprehensive error handling for network issues

### 📊 **Streamlit Dashboard**
- **Real-time Processing**: Live progress tracking
- **Interactive Thresholds**: Adjustable similarity thresholds
- **Visual Results**: DataFrame displays with filtering
- **Export Capabilities**: CSV export for further analysis
- **Failed URL Tracking**: Detailed error reporting

## Installation

### Local Setup

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/duplicate-content-detector.git
cd duplicate-content-detector
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. **Run the application**:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. **Fork this repository**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Select your forked repository**
5. **Deploy!**

## Usage

### Method 1: Manual URL Input
Enter URLs one per line:
```
https://example.com/page1
https://example.com/page2
https://example.com/page3
```

### Method 2: Sitemap URL
Provide a sitemap URL to automatically extract all URLs:
```
https://example.com/sitemap.xml
```

### Method 3: Bulk Text Input
Paste URLs separated by commas or newlines:
```
https://example.com/page1, https://example.com/page2
```

## Configuration Options

### Similarity Threshold
- **Range**: 0.0 - 1.0
- **Default**: 0.75
- **Recommendation**: 0.7-0.85 for most use cases

### Minimum Content Length
- **Range**: 50 - 1000 words
- **Default**: 100 words
- **Purpose**: Filter out pages with insufficient content

### Rate Limiting
- **Default Delay**: 2 seconds between requests
- **Purpose**: Prevent 429 errors from target servers

## Architecture

### Modular Design
```
duplicate_content_detector/
├── app.py              # Streamlit interface
├── config.py           # Configuration management
├── scraper.py          # Web scraping engine
├── detector.py         # AI/NLP detection engine
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

### Detection Pipeline
1. **URL Collection**: Manual input or sitemap extraction
2. **Content Scraping**: Intelligent content extraction
3. **Preprocessing**: Text cleaning and normalization
4. **Similarity Calculation**: Multiple AI/NLP methods
5. **Result Aggregation**: Weighted scoring system
6. **Visualization**: Interactive Streamlit dashboard

## AI Models Used

### Sentence Transformers
- **Model**: `all-MiniLM-L6-v2`
- **Purpose**: Semantic similarity detection
- **Size**: ~50MB
- **Performance**: Fast and accurate for most use cases

### BERTScore
- **Model**: `roberta-large`
- **Purpose**: Contextual similarity scoring
- **Note**: May show warnings about uninitialized weights (normal behavior)

## Performance Optimization

### Caching Strategy
- **Content Cache**: 1-hour TTL for scraped content
- **Model Cache**: Persistent model loading
- **Result Cache**: Session-based caching

### Rate Limiting
- **Request Delay**: Configurable between requests
- **Connection Pool**: Limited to prevent overwhelming servers
- **Retry Logic**: Exponential backoff for failed requests

## Error Handling

### Common Issues & Solutions

**429 Too Many Requests**
- **Solution**: Increase rate limiting delay in settings
- **Alternative**: Use smaller batch sizes

**Connection Timeouts**
- **Solution**: Increase timeout values in config
- **Check**: Target website availability

**Empty Results**
- **Solution**: Lower minimum content length threshold
- **Check**: URL accessibility and content availability

## Output Format

### CSV Export Structure
| Column | Description |
|--------|-------------|
| URL 1 | First URL in comparison |
| URL 2 | Second URL in comparison |
| Similarity | Overall similarity score (0-1) |
| Confidence | Detection confidence level |
| Is Duplicate | Boolean flag for duplicates |
| Common Content | Shared content between pages |

### Example Output
```
URL 1,URL 2,Similarity,Confidence,Is Duplicate,Common Content
https://example.com/page1,https://example.com/page2,0.85,0.92,Yes,"Both pages discuss..."
```

## Advanced Usage

### Custom Configuration
```python
from config import Config

# Adjust settings
Config.SEMANTIC_THRESHOLD = 0.8
Config.MIN_CONTENT_LENGTH = 150
Config.MAX_WORKERS = 5
```

### Batch Processing
For large websites (>1000 URLs), consider:
1. Processing in smaller batches
2. Using sitemap.xml for URL discovery
3. Implementing custom rate limiting

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Commit changes**: `git commit -am 'Add new feature'`
4. **Push to branch**: `git push origin feature/new-feature`
5. **Create a Pull Request**

## License

MIT License - see LICENSE file for details

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/duplicate-content-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/duplicate-content-detector/discussions)
- **Email**: your.email@example.com

## Roadmap

- [ ] Multi-language support
- [ ] Real-time similarity alerts
- [ ] API endpoints for integration
- [ ] Advanced visualization charts
- [ ] Cloud storage integration
- [ ] Scheduled monitoring
- [ ] Team collaboration features

## Acknowledgments

- **Streamlit** for the amazing framework
- **Hugging Face** for transformer models
- **NLTK** for natural language processing tools
- **BeautifulSoup** for HTML parsing
- **Sentence Transformers** for semantic embeddings
