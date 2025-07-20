"""Streamlit application for duplicate content detection."""

import streamlit as st
import pandas as pd
import asyncio
import logging
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import io
from config import Config
from scraper import WebScraper, ScrapedContent
from detector import DuplicateDetector, DuplicateResult

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
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .duplicate-card {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .safe-card {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.config = Config()
        self.detector = DuplicateDetector(self.config)
        
    def run(self):
        """Run the Streamlit application."""
        st.title("🔍 AI-Powered Duplicate Content Detector")
        st.markdown("### Advanced NLP-based duplicate detection for websites")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # URL input
            url_input = st.text_area(
                "Enter URLs (one per line)",
                placeholder="https://example.com/page1\nhttps://example.com/page2",
                height=100
            )
            
            # Sitemap input
            sitemap_url = st.text_input(
                "Or enter sitemap URL",
                placeholder="https://example.com/sitemap.xml"
            )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                self.config.COSINE_THRESHOLD = st.slider(
                    "Cosine Similarity Threshold",
                    0.0, 1.0, self.config.COSINE_THRESHOLD
                )
                self.config.SEMANTIC_THRESHOLD = st.slider(
                    "Semantic Similarity Threshold",
                    0.0, 1.0, self.config.SEMANTIC_THRESHOLD
                )
                self.config.MIN_CONTENT_LENGTH = st.number_input(
                    "Minimum Content Length",
                    min_value=50, max_value=1000, value=self.config.MIN_CONTENT_LENGTH
                )
                self.config.MAX_WORKERS = st.number_input(
                    "Max Concurrent Workers",
                    min_value=1, max_value=20, value=self.config.MAX_WORKERS
                )
                
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                analyze_button = st.button("🔍 Analyze", type="primary")
            with col2:
                clear_button = st.button("🗑️ Clear Results")
                
        # Main content area
        if analyze_button:
            self.analyze_content(url_input, sitemap_url)
        elif clear_button:
            st.session_state.clear()
            st.rerun()
            
        # Display results if available
        if 'results' in st.session_state:
            self.display_results()
            
    def analyze_content(self, url_input: str, sitemap_url: str):
        """Analyze content for duplicates."""
        with st.spinner("🔄 Scraping and analyzing content..."):
            try:
                # Collect URLs
                urls = self.collect_urls(url_input, sitemap_url)
                
                if not urls:
                    st.error("❌ No valid URLs provided")
                    return
                    
                st.info(f"📊 Analyzing {len(urls)} URLs...")
                
                # Scrape content
                contents = asyncio.run(self.scrape_all_urls(urls))
                
                if not contents:
                    st.error("❌ No content could be scraped")
                    return
                    
                # Detect duplicates
                results = self.detector.detect_duplicates(contents)
                
                # Store results
                st.session_state['results'] = results
                st.session_state['contents'] = contents
                st.session_state['urls'] = urls
                
                st.success(f"✅ Analysis complete! Found {len(results)} potential duplicates")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                logger.exception("Analysis error")
                
    def collect_urls(self, url_input: str, sitemap_url: str) -> List[str]:
        """Collect URLs from input and sitemap."""
        urls = []
        
        # Parse manual URLs
        if url_input:
            manual_urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            urls.extend(manual_urls)
            
        # Parse sitemap
        if sitemap_url:
            try:
                scraper = WebScraper(self.config)
                sitemap_urls = scraper.extract_sitemap_urls(sitemap_url)
                urls.extend(sitemap_urls)
                st.info(f"📋 Added {len(sitemap_urls)} URLs from sitemap")
            except Exception as e:
                st.warning(f"⚠️ Could not parse sitemap: {e}")
                
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = [url for url in urls if not (url in seen or seen.add(url))]
        
        return unique_urls
        
    async def scrape_all_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """Scrape all URLs concurrently."""
        async with WebScraper(self.config) as scraper:
            return await scraper.scrape_urls(urls)
            
    def display_results(self):
        """Display analysis results."""
        results = st.session_state['results']
        contents = st.session_state['contents']
        
        if not results:
            st.info("✅ No duplicates found!")
            return
            
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total URLs", len(st.session_state['urls']))
        with col2:
            st.metric("Content Scraped", len(contents))
        with col3:
            duplicates = sum(1 for r in results if r.is_duplicate)
            st.metric("Duplicates Found", duplicates)
        with col4:
            avg_similarity = np.mean([r.similarity_score for r in results]) if results else 0
            st.metric("Avg Similarity", f"{avg_similarity:.2%}")
            
        # Visualizations
        self.create_visualizations(results)
        
        # Detailed results
        st.header("📋 Detailed Results")
        
        # Filter duplicates
        show_duplicates = st.checkbox("Show only confirmed duplicates", value=True)
        
        filtered_results = [r for r in results if not show_duplicates or r.is_duplicate]
        
        for idx, result in enumerate(filtered_results):
            self.display_result_card(result, idx)
            
    def create_visualizations(self, results: List[DuplicateResult]):
        """Create interactive visualizations."""
        st.header("📊 Analysis Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Similarity distribution
            similarities = [r.similarity_score for r in results]
            fig = px.histogram(
                x=similarities,
                nbins=20,
                title="Similarity Score Distribution",
                labels={'x': 'Similarity Score', 'y': 'Count'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Duplicate status pie chart
            duplicate_counts = {
                'Duplicates': sum(1 for r in results if r.is_duplicate),
                'Non-duplicates': sum(1 for r in results if not r.is_duplicate)
            }
            fig = px.pie(
                values=list(duplicate_counts.values()),
                names=list(duplicate_counts.keys()),
                title="Duplicate Status Distribution"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        # Similarity heatmap
        if len(results) <= 50:  # Limit for performance
            self.create_similarity_heatmap(results)
            
    def create_similarity_heatmap(self, results: List[DuplicateResult]):
        """Create similarity heatmap."""
        st.subheader("🔥 Similarity Heatmap")
        
        # Get unique URLs
        urls = list(set([r.url1 for r in results] + [r.url2 for r in results]))
        url_to_idx = {url: idx for idx, url in enumerate(urls)}
        
        # Create matrix
        matrix = np.zeros((len(urls), len(urls)))
        for result in results:
            i = url_to_idx[result.url1]
            j = url_to_idx[result.url2]
            matrix[i, j] = result.similarity_score
            matrix[j, i] = result.similarity_score
            
        # Create heatmap
        fig = px.imshow(
            matrix,
            labels=dict(x="URL", y="URL", color="Similarity"),
            x=urls,
            y=urls,
            color_continuous_scale="RdYlBu_r"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    def display_result_card(self, result: DuplicateResult, index: int):
        """Display individual result card."""
        card_class = "duplicate-card" if result.is_duplicate else "safe-card"
        
        with st.expander(
            f"{'🔴' if result.is_duplicate else '🟢'} "
            f"Pair {index + 1}: {result.similarity_score:.2%} similarity",
            expanded=result.is_duplicate
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**URL 1:**")
                st.code(result.url1, language=None)
                st.markdown("**URL 2:**")
                st.code(result.url2, language=None)
                
            with col2:
                st.metric("Similarity Score", f"{result.similarity_score:.2%}")
                st.metric("Confidence", f"{result.confidence:.2%}")
                
            # Detailed scores
            with st.expander("📊 Detailed Similarity Scores"):
                scores = result.metadata.get('individual_scores', {})
                if scores:
                    df = pd.DataFrame(
                        list(scores.items()),
                        columns=['Method', 'Score']
                    )
                    st.dataframe(df, use_container_width=True)
                    
            # Common content
            if result.common_content:
                with st.expander("📝 Common Content"):
                    st.text_area("Common text", result.common_content[:500] + "...", height=100)
                    
            # Differences
            try:
                differences = json.loads(result.differences)
                if differences.get('added') or differences.get('removed'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Added content:**")
                        st.text_area("", differences.get('added', '')[:200] + "...", height=80)
                    with col2:
                        st.markdown("**Removed content:**")
                        st.text_area("", differences.get('removed', '')[:200] + "...", height=80)
            except:
                pass
                
    def export_results(self):
        """Export results functionality."""
        if 'results' not in st.session_state:
            return
            
        results = st.session_state['results']
        
        # Prepare data for export
        export_data = []
        for result in results:
            export_data.append({
                'URL1': result.url1,
                'URL2': result.url2,
                'Similarity Score': result.similarity_score,
                'Confidence': result.confidence,
                'Is Duplicate': result.is_duplicate,
                'Common Content': result.common_content,
                'Differences': result.differences,
                **result.metadata.get('individual_scores', {})
            })
            
        df = pd.DataFrame(export_data)
        
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"duplicate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # JSON export
        json_data = json.dumps([r.__dict__ for r in results], indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"duplicate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
    
    # Add export section at the bottom
    st.divider()
    st.header("📤 Export Results")
    app.export_results()
