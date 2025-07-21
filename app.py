"""Enhanced Streamlit app for duplicate content detection with full functionality."""

import streamlit as st
import logging
import pandas as pd
import numpy as np
import time
import requests
import asyncio
from typing import List, Tuple, Optional
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import validators
from config import Config
from scraper import WebScraper, ScrapedContent
from detector import DuplicateDetector, DuplicateResult
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Duplicate Content Detector",
    page_icon="🔍",
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
    .url-input { min-height: 120px; }
    .error-message { color: #ff4b4b; font-size: 0.8em; }
    .success-message { color: #00c851; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_detector():
    """Initialize the duplicate detector with caching."""
    return DuplicateDetector(Config())


@st.cache_resource
def init_scraper():
    """Initialize the web scraper."""
    return WebScraper(Config())


def create_session_with_retries() -> requests.Session:
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def extract_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Extract URLs from XML sitemap with comprehensive error handling."""
    try:
        if not sitemap_url.strip():
            return []
            
        session = create_session_with_retries()
        response = session.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = []
        
        # Handle regular sitemap
        for loc in soup.find_all('loc'):
            url = loc.text.strip()
            if validate_url(url):
                urls.append(url)
        
        # Handle sitemap index
        for sitemap in soup.find_all('sitemap'):
            loc = sitemap.find('loc')
            if loc:
                nested_urls = extract_urls_from_sitemap(loc.text.strip())
                urls.extend(nested_urls)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = [url for url in urls if not (url in seen or seen.add(url))]
        
        return unique_urls
        
    except Exception as e:
        st.error(f"Error extracting sitemap: {str(e)}")
        logger.error(f"Sitemap extraction error: {e}")
        return []


def validate_url(url: str) -> bool:
    """Validate URL format with comprehensive checks."""
    try:
        if not url or not isinstance(url, str):
            return False
            
        url = url.strip()
        if not url:
            return False
            
        result = urlparse(url)
        return all([
            result.scheme in ['http', 'https'],
            result.netloc,
            len(result.netloc) > 3,
            '.' in result.netloc
        ])
    except Exception:
        return False


def extract_content_from_url(url: str, session: requests.Session) -> Tuple[Optional[ScrapedContent], Optional[str]]:
    """Extract content from a single URL with detailed error reporting."""
    try:
        if not validate_url(url):
            return None, "Invalid URL format"
        
        response = session.get(
            url,
            timeout=30,
            headers={
                'User-Agent': Config.USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return None, f"Invalid content type: {content_type}"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
            element.decompose()
        
        # Extract title
        title = soup.find('title')
        title = title.get_text(strip=True) if title else ""
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_desc.get('content', '')[:500] if meta_desc else ""
        
        # Extract main content with multiple strategies
        content = ""
        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.main-content',
            '.post-content', '.entry-content', '#content', '#main-content',
            '.article-content', '.post-body', '.entry-summary'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                if len(content) > Config.MIN_CONTENT_LENGTH:
                    break
        
        # Fallback to body content
        if not content or len(content) < Config.MIN_CONTENT_LENGTH:
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)
        
        # Clean content
        content = ' '.join(content.split())
        
        if len(content) < Config.MIN_CONTENT_LENGTH:
            return None, f"Content too short ({len(content)} chars)"
        
        # Extract headings
        headings = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            headings.extend([h.get_text(strip=True) for h in soup.find_all(tag)])
        
        # Extract images and links
        images = [img.get('src') for img in soup.find_all('img') if img.get('src')]
        links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
        links = [urljoin(url, link) for link in links]
        
        return ScrapedContent(
            url=url,
            title=title,
            content=content,
            meta_description=meta_description,
            headings=headings,
            word_count=len(content.split()),
            language="en",
            images=images,
            links=links
        ), None
        
    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.ConnectionError:
        return None, "Connection error"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {e.response.status_code}"
    except Exception as e:
        return None, str(e)


def process_urls_concurrent(urls: List[str], progress_bar, status_text) -> List[ScrapedContent]:
    """Process URLs concurrently with rate limiting and error handling."""
    contents = []
    failed_urls = []
    
    # Create session with retries
    session = create_session_with_retries()
    
    # Rate limiting configuration
    max_workers = min(Config.MAX_WORKERS, 10)
    delay_between_batches = 0.5
    
    # Process URLs in batches
    total_urls = len(urls)
    processed = 0
    
    for i in range(0, total_urls, max_workers):
        batch_urls = urls[i:i + max_workers]
        
        # Process batch concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(extract_content_from_url, url, session): url 
                for url in batch_urls
            }
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                processed += 1
                
                try:
                    content, error = future.result()
                    if content:
                        contents.append(content)
                    else:
                        failed_urls.append((url, error))
                        
                except Exception as e:
                    failed_urls.append((url, str(e)))
                
                # Update progress
                progress = processed / total_urls
                progress_bar.progress(progress)
                status_text.text(f"Processing URL {processed}/{total_urls}")
        
        # Rate limiting between batches
        if i + max_workers < total_urls:
            time.sleep(delay_between_batches)
    
    # Display failed URLs
    if failed_urls:
        with st.sidebar.expander(f"Failed URLs ({len(failed_urls)})"):
            for url, error in failed_urls:
                st.text(f"❌ {url}")
                st.caption(error)
    
    return contents


def main():
    """Main application with full functionality."""
    st.title("🔍 AI-Powered Duplicate Content Detector")
    st.markdown("Detect duplicate content using advanced AI/NLP techniques")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Manual URLs", "Sitemap URL", "Text Input", "File Upload"],
            help="Choose how to provide URLs for analysis"
        )
        
        urls = []
        
        # URL input based on method
        if input_method == "Manual URLs":
            url_input = st.text_area(
                "Enter URLs (one per line)",
                placeholder="https://example.com/page1\nhttps://example.com/page2",
                height=150,
                key="manual_urls"
            )
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            
        elif input_method == "Sitemap URL":
            sitemap_url = st.text_input(
                "Sitemap URL",
                placeholder="https://example.com/sitemap.xml",
                help="Enter the full URL to your XML sitemap"
            )
            if sitemap_url:
                with st.spinner("Extracting URLs from sitemap..."):
                    urls = extract_urls_from_sitemap(sitemap_url)
                    st.success(f"Found {len(urls)} URLs in sitemap")
                    
        elif input_method == "Text Input":
            text_input = st.text_area(
                "Enter URLs separated by commas or newlines",
                placeholder="https://example.com/page1, https://example.com/page2",
                height=150,
                key="text_urls"
            )
            urls = [url.strip() for url in text_input.replace(',', '\n').split('\n') if url.strip()]
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload a text file with URLs",
                type=['txt', 'csv'],
                help="Upload a file with one URL per line"
            )
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                urls = [url.strip() for url in content.split('\n') if url.strip()]
        
        # Advanced settings
        with st.expander("Advanced Settings", expanded=False):
            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=Config.SEMANTIC_THRESHOLD,
                step=0.01,
                help="Lower values detect more duplicates"
            )
            
            min_content_length = st.number_input(
                "Minimum Content Length (words)",
                min_value=50,
                max_value=1000,
                value=Config.MIN_CONTENT_LENGTH,
                step=10,
                help="Filter out pages with insufficient content"
            )
            
            max_workers = st.number_input(
                "Max Concurrent Requests",
                min_value=1,
                max_value=20,
                value=Config.MAX_WORKERS,
                step=1,
                help="Higher values process faster but may hit rate limits"
            )
        
        # Update configuration
        Config.SEMANTIC_THRESHOLD = threshold
        Config.MIN_CONTENT_LENGTH = min_content_length
        Config.MAX_WORKERS = max_workers
    
    # Main content area
    if urls:
        st.header("Processing URLs")
        
        # Display URL statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total URLs", len(urls))
        with col2:
            st.metric("Threshold", f"{Config.SEMANTIC_THRESHOLD:.2f}")
        with col3:
            st.metric("Min Length", f"{Config.MIN_CONTENT_LENGTH} words")
        with col4:
            st.metric("Workers", Config.MAX_WORKERS)
        
        # Process URLs
        if st.button("Start Analysis", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Validate URLs
                valid_urls = [url for url in urls if validate_url(url)]
                invalid_urls = [url for url in urls if not validate_url(url)]
                
                if invalid_urls:
                    st.warning(f"Found {len(invalid_urls)} invalid URLs")
                
                if not valid_urls:
                    st.error("No valid URLs to process")
                    return
                
                # Scrape content
                contents = process_urls_concurrent(valid_urls, progress_bar, status_text)
                
                if not contents:
                    st.error("No content could be extracted. Please check the URLs and try again.")
                    return
                
                st.success(f"Successfully extracted content from {len(contents)} URLs")
                
                # Detect duplicates
                with st.spinner("Analyzing content for duplicates..."):
                    detector = init_detector()
                    results = detector.detect_duplicates(contents)
                
                # Display results
                st.header("Analysis Results")
                
                if results:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Pairs", len(results))
                    with col2:
                        duplicates = [r for r in results if r.is_duplicate]
                        st.metric("Duplicates Found", len(duplicates))
                    with col3:
                        avg_similarity = np.mean([r.similarity_score for r in results])
                        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                    with col4:
                        max_similarity = max([r.similarity_score for r in results]) if results else 0
                        st.metric("Max Similarity", f"{max_similarity:.3f}")
                    
                    # Results visualization
                    st.subheader("Detailed Results")
                    
                    # Create DataFrame
                    df_data = []
                    for result in results:
                        df_data.append({
                            'URL 1': result.url1,
                            'URL 2': result.url2,
                            'Similarity': f"{result.similarity_score:.3f}",
                            'Confidence': f"{result.confidence:.2f}",
                            'Is Duplicate': 'Yes' if result.is_duplicate else 'No',
                            'Common Content': result.common_content[:300] + '...' 
                            if len(result.common_content) > 300 else result.common_content
                        })
                    
                    df = pd.DataFrame(df_data)
                    
                    # Filter duplicates
                    duplicate_df = df[df['Is Duplicate'] == 'Yes']
                    
                    if not duplicate_df.empty:
                        st.warning(f"🚨 Found {len(duplicate_df)} duplicate pairs above threshold")
                        
                        # Display duplicates prominently
                        st.subheader("🎯 Duplicate Pairs")
                        st.dataframe(duplicate_df, use_container_width=True)
                        
                        # Similarity distribution
                        st.subheader("📊 Similarity Distribution")
                        similarity_scores = [r.similarity_score for r in results]
                        st.bar_chart(pd.Series(similarity_scores))
                        
                    else:
                        st.info("✅ No duplicates found above the threshold")
                    
                    # Export functionality
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Results (CSV)",
                            data=csv,
                            file_name="duplicate_content_analysis.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        if not duplicate_df.empty:
                            duplicate_csv = duplicate_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Duplicates Only (CSV)",
                                data=duplicate_csv,
                                file_name="duplicates_only.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                else:
                    st.info("No duplicate pairs found")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                logger.exception("Analysis error")
    
    else:
        st.info("👆 Please enter URLs to analyze using one of the input methods above")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, AI/NLP techniques, and ❤️*")


if __name__ == "__main__":
    main()
