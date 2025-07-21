"""Enterprise-grade duplicate content detector with blazing fast performance."""

import streamlit as st
import pandas as pd
import json
import logging
import time
from datetime import datetime
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enterprise Duplicate Content Detector",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stProgress .st-bo { background-color: #1f77b4; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config() -> Config:
    """Load configuration."""
    return Config()


@st.cache_data
def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Load dataframe from uploaded file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


class FastScraper:
    """Ultra-fast scraper using ThreadPoolExecutor."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
    def extract_content(self, url: str) -> dict:
        """Extract content from URL using fast extraction."""
        try:
            response = self.session.get(
                url, 
                timeout=self.config.REQUEST_TIMEOUT,
                allow_redirects=True
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title = title.get_text(strip=True) if title else ""
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_desc.get('content', '') if meta_desc else ""
            
            # Extract main content
            content_selectors = [
                'main', 'article', '[role="main"]', '.content', '.main-content',
                '.post-content', '.entry-content', '#content', '#main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if len(text) > len(content):
                        content = text
                        break
                    
            if not content:
                content = soup.get_text(separator=' ', strip=True)
            
            # Clean content
            content = re.sub(r'\s+', ' ', content).strip()
            
            if len(content) < self.config.MIN_CONTENT_LENGTH:
                return None
            
            # Extract links
            links = [urljoin(url, a.get('href')) for a in soup.find_all('a') if a.get('href')]
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'meta_description': meta_description,
                'word_count': len(content.split()),
                'links': links[:10]  # Limit to first 10 links
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract {url}: {e}")
            return None
    
    def scrape_batch(self, urls: list, progress_callback=None) -> list:
        """Scrape URLs in parallel batches."""
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_url = {executor.submit(self.extract_content, url): url for url in urls}
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(urls), url)
                        
                except Exception as e:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(urls), url)
                    logger.warning(f"Error processing {url}: {e}")
        
        return results


class FastDuplicateDetector:
    """Ultra-fast duplicate detection using optimized algorithms."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def calculate_jaccard_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculate Jaccard similarity using n-grams."""
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            return {' '.join(words[i:i+n]) for i in range(len(words)-n+1)}
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_tfidf_similarity(self, texts: list) -> np.ndarray:
        """Calculate TF-IDF similarity matrix."""
        if len(texts) < 2:
            return np.array([])
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    def detect_duplicates_fast(self, contents: list) -> list:
        """Fast duplicate detection using optimized algorithms."""
        if len(contents) < 2:
            return []
        
        # Extract texts
        texts = [item['content'] for item in contents]
        urls = [item['url'] for item in contents]
        
        # Calculate TF-IDF similarity matrix
        similarity_matrix = self.calculate_tfidf_similarity(texts)
        
        # Find duplicates
        duplicates = []
        n = len(contents)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.config.SEMANTIC_THRESHOLD:
                    # Calculate additional metrics for confidence
                    jaccard_sim = self.calculate_jaccard_similarity(texts[i], texts[j])
                    
                    # Weighted similarity score
                    final_score = (similarity * 0.7 + jaccard_sim * 0.3)
                    
                    if final_score >= self.config.SEMANTIC_THRESHOLD:
                        # Find common content
                        matcher = SequenceMatcher(None, texts[i], texts[j])
                        common = []
                        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                            if tag == 'equal':
                                common.append(texts[i][i1:i2])
                        
                        common_content = ' '.join(common)[:500] + '...' if common else ""
                        
                        duplicates.append({
                            'url1': urls[i],
                            'url2': urls[j],
                            'similarity_score': final_score,
                            'confidence': min(1.0, final_score * 1.1),
                            'common_content': common_content,
                            'is_duplicate': True
                        })
        
        return sorted(duplicates, key=lambda x: x['similarity_score'], reverse=True)


def extract_sitemap_urls(sitemap_url: str) -> list:
    """Extract URLs from XML sitemap."""
    try:
        import requests
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = []
        
        # Handle regular sitemap
        for loc in soup.find_all('loc'):
            urls.append(loc.text.strip())
            
        # Handle sitemap index
        for sitemap in soup.find_all('sitemap'):
            loc = sitemap.find('loc')
            if loc:
                nested_urls = extract_sitemap_urls(loc.text.strip())
                urls.extend(nested_urls)
                
        return urls
        
    except Exception as e:
        logger.error(f"Error extracting sitemap URLs: {e}")
        return []


def analyze_urls_fast(urls: list, config: Config, progress_callback=None) -> dict:
    """Ultra-fast analysis using ThreadPoolExecutor."""
    start_time = time.time()
    
    # Phase 1: Fast scraping
    scraper = FastScraper(config)
    contents = scraper.scrape_batch(urls, progress_callback)
    
    if not contents:
        return {"error": "No content could be scraped", "contents": [], "results": []}
    
    # Phase 2: Fast duplicate detection
    detector = FastDuplicateDetector(config)
    duplicates = detector.detect_duplicates_fast(contents)
    
    total_time = time.time() - start_time
    
    return {
        "contents": contents,
        "results": duplicates,
        "summary": {
            "total_urls": len(urls),
            "content_scraped": len(contents),
            "duplicates_found": len(duplicates),
            "similarity_threshold": config.SEMANTIC_THRESHOLD,
            "processing_time": total_time,
            "urls_per_second": len(urls) / total_time
        }
    }


def main():
    """Main application."""
    st.title("🚀 Enterprise Duplicate Content Detector")
    st.markdown("Ultra-fast duplicate content detection for large-scale websites")
    
    st.sidebar.header("📋 Input Options")
    
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Paste URLs", "Upload File", "Sitemap URL"]
    )
    
    urls = []
    
    if input_method == "Paste URLs":
        url_input = st.sidebar.text_area(
            "Enter URLs (one per line):",
            height=200,
            placeholder="https://example.com/page1\nhttps://example.com/page2"
        )
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        
    elif input_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing URLs. The app will detect URL columns automatically."
        )
        
        if uploaded_file is not None:
            df = load_dataframe(uploaded_file)
            if df is not None:
                columns = df.columns.tolist()
                url_column = st.selectbox("Select the column containing URLs:", columns)
                
                if url_column:
                    urls = df[url_column].dropna().astype(str).tolist()
                    urls = [url.strip() for url in urls if url.strip()]
                    st.sidebar.success(f"✅ Found {len(urls)} URLs")
                
    elif input_method == "Sitemap URL":
        sitemap_url = st.sidebar.text_input(
            "Enter sitemap URL:",
            placeholder="https://example.com/sitemap.xml"
        )
        if sitemap_url:
            urls = extract_sitemap_urls(sitemap_url)
            if urls:
                st.sidebar.success(f"✅ Found {len(urls)} URLs from sitemap")
    
    st.sidebar.header("⚙️ Configuration")
    config = load_config()
    
    config.SEMANTIC_THRESHOLD = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Lower values detect more duplicates"
    )
    
    config.MAX_WORKERS = st.sidebar.slider(
        "Max Concurrent Workers",
        min_value=1,
        max_value=50,
        value=25,
        step=1,
        help="Higher values process faster but may hit rate limits"
    )
    
    config.REQUEST_TIMEOUT = st.sidebar.slider(
        "Request Timeout (seconds)",
        min_value=5,
        max_value=60,
        value=15,
        step=5
    )
    
    if st.sidebar.button("🚀 Start Fast Analysis", type="primary", disabled=not urls):
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        speed_text = st.empty()
        
        def update_progress(current: int, total: int, message: str):
            """Update progress with real-time feedback."""
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(message)
            
            if current > 0 and total > 0:
                elapsed = time.time() - start_time
                speed = current / elapsed
                speed_text.text(f"⚡ {current}/{total} URLs ({speed:.1f} URLs/sec)")
        
        start_time = time.time()
        
        with st.spinner("🔄 Processing..."):
            results = analyze_urls_fast(urls, config, update_progress)
        
        # Final update
        progress_bar.progress(1.0)
        status_text.text("✅ Complete!")
        speed_text.text(f"🎯 Total time: {results['summary']['processing_time']:.1f}s")
        
        # Display results
        st.header("⚡ Enterprise Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total URLs", results["summary"]["total_urls"])
        with col2:
            st.metric("Content Scraped", results["summary"]["content_scraped"])
        with col3:
            st.metric("Duplicates Found", results["summary"]["duplicates_found"])
        with col4:
            st.metric("Speed", f"{results['summary']['urls_per_second']:.1f} URLs/sec")
        
        if results["summary"]["urls_per_second"] > 5:
            st.success("🚀 **Enterprise Performance Achieved!**")
        
        duplicates = results["results"]
        
        if duplicates:
            st.header("🔍 Fast Duplicate Results")
            
            # Display in a clean table
            df_display = pd.DataFrame([
                {
                    "URL 1": d["url1"],
                    "URL 2": d["url2"],
                    "Similarity": f"{d['similarity_score']:.1%}",
                    "Confidence": f"{d['confidence']:.1%}"
                }
                for d in duplicates
            ])
            
            st.dataframe(df_display, use_container_width=True)
            
            # Export options
            st.header("📥 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"enterprise_duplicate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = json.dumps({
                    "summary": results["summary"],
                    "duplicates": duplicates
                }, indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"enterprise_duplicate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.success("✅ No duplicates found above the threshold!")


if __name__ == "__main__":
    main()
