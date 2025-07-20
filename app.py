"""Streamlit app for AI-powered duplicate content detection."""

import streamlit as st
import asyncio
import pandas as pd
import json
import logging
from datetime import datetime
import plotly.express as px
from scraper import WebScraper
from detector import DuplicateDetector
from config import Config

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


class ProgressTracker:
    """Track progress for scraping and analysis."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(self, message: str, step_increment: int = 1):
        """Update progress."""
        self.current_step += step_increment
        progress = min(self.current_step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        self.status_text.text(message)
        
    def complete(self):
        """Mark as complete."""
        self.progress_bar.progress(1.0)
        self.status_text.text("✅ Complete!")


async def analyze_urls_with_progress(urls: list, config: Config) -> dict:
    """Analyze URLs with detailed progress tracking."""
    total_steps = len(urls) + 3
    
    progress = ProgressTracker(total_steps)
    
    try:
        progress.update("🚀 Initializing scraper...")
        
        progress.update("📄 Scraping content from URLs...")
        async with WebScraper(config) as scraper:
            contents = await scraper.scrape_urls(urls)
        
        if not contents:
            return {"error": "No content could be scraped from the provided URLs"}
        
        for content in contents:
            progress.update(f"✅ Scraped: {content.url} ({content.word_count} words)", 0)
        
        progress.update("🔍 Analyzing content for duplicates...")
        detector = DuplicateDetector(config)
        results = detector.detect_duplicates(contents)
        
        progress.update("📊 Generating summary report...")
        duplicates = [r for r in results if r.is_duplicate]
        
        summary = {
            "total_urls": len(urls),
            "content_scraped": len(contents),
            "duplicates_found": len(duplicates),
            "similarity_threshold": config.SEMANTIC_THRESHOLD,
            "analysis_date": datetime.now().isoformat()
        }
        
        progress.complete()
        
        return {
            "contents": contents,
            "results": results,
            "summary": summary
        }
        
    except Exception as e:
        progress.status_text.error(f"❌ Error: {str(e)}")
        return {"error": str(e)}


def display_results(results: dict):
    """Display analysis results."""
    if "error" in results:
        st.error(f"Analysis failed: {results['error']}")
        return
    
    st.header("📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total URLs", results["summary"]["total_urls"])
    with col2:
        st.metric("Content Scraped", results["summary"]["content_scraped"])
    with col3:
        st.metric("Duplicates Found", results["summary"]["duplicates_found"])
    with col4:
        st.metric("Threshold", f"{results['summary']['similarity_threshold']:.0%}")
    
    duplicates = [r for r in results["results"] if r.is_duplicate]
    
    if duplicates:
        st.header("🔍 Duplicate Results")
        duplicates = sorted(duplicates, key=lambda x: x.similarity_score, reverse=True)
        
        for i, result in enumerate(duplicates):
            with st.expander(f"Duplicate Pair {i+1} ({result.similarity_score:.1%} similarity)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**URL 1:** {result.url1}")
                    st.markdown(f"**URL 2:** {result.url2}")
                    st.markdown(f"**Confidence:** {result.confidence:.1%}")
                
                with col2:
                    st.markdown("**Common Content:**")
                    st.text_area("", result.common_content, height=100, key=f"common_{i}")
        
        st.header("📥 Export Results")
        csv_data = []
        for result in duplicates:
            csv_data.append({
                "URL 1": result.url1,
                "URL 2": result.url2,
                "Similarity Score": f"{result.similarity_score:.2%}",
                "Confidence": f"{result.confidence:.2%}",
                "Common Content": result.common_content
            })
        
        df = pd.DataFrame(csv_data)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"duplicate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = json.dumps({
                "summary": results["summary"],
                "duplicates": csv_data
            }, indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"duplicate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.success("✅ No duplicates found above the threshold!")


def create_visualizations(results: dict):
    """Create interactive visualizations."""
    if "error" in results or not results["results"]:
        return
    
    st.header("📈 Visualizations")
    
    similarities = [r.similarity_score for r in results["results"]]
    
    fig_hist = px.histogram(
        x=similarities,
        nbins=20,
        title="Similarity Score Distribution",
        labels={"x": "Similarity Score", "y": "Count"},
        color_discrete_sequence=["#1f77b4"]
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    duplicates = [r for r in results["results"] if r.is_duplicate]
    if len(duplicates) > 1:
        urls = list(set([d.url1 for d in duplicates] + [d.url2 for d in duplicates]))
        if len(urls) <= 10:
            similarity_matrix = []
            for url1 in urls:
                row = []
                for url2 in urls:
                    if url1 == url2:
                        row.append(1.0)
                    else:
                        score = 0.0
                        for d in duplicates:
                            if (d.url1 == url1 and d.url2 == url2) or (d.url1 == url2 and d.url2 == url1):
                                score = d.similarity_score
                                break
                        row.append(score)
                similarity_matrix.append(row)
            
            fig_heatmap = px.imshow(
                similarity_matrix,
                x=urls,
                y=urls,
                title="Similarity Heatmap",
                color_continuous_scale="Blues",
                aspect="auto"
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)


def main():
    """Main application."""
    st.title("🔍 AI-Powered Duplicate Content Detector")
    st.markdown("Detect duplicate content across websites using advanced AI/NLP techniques")
    
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
            config = load_config()
            scraper = WebScraper(config)
            urls = scraper.extract_sitemap_urls(sitemap_url)
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
        max_value=20,
        value=10,
        step=1,
        help="Higher values process faster but may hit rate limits"
    )
    
    if st.sidebar.button("🚀 Start Analysis", type="primary", disabled=not urls):
        with st.spinner("🔄 Analyzing..."):
            results = asyncio.run(analyze_urls_with_progress(urls, config))
            
            if "error" not in results:
                display_results(results)
                create_visualizations(results)
            else:
                st.error(f"❌ Analysis failed: {results['error']}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ using Streamlit and AI")


if __name__ == "__main__":
    main()
