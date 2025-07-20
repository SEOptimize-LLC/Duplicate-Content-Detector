#!/usr/bin/env python3
"""Test script to verify the duplicate content detector system."""

import asyncio
import sys
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from scraper import WebScraper, ScrapedContent
from detector import DuplicateDetector, DuplicateResult
from utils import DuplicateAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scraper():
    """Test the web scraper."""
    print("🧪 Testing Web Scraper...")
    
    config = Config()
    config.MAX_WORKERS = 3
    config.MIN_CONTENT_LENGTH = 50
    
    test_urls = [
        "https://httpbin.org/html",
        "https://example.com",
    ]
    
    try:
        async with WebScraper(config) as scraper:
            contents = await scraper.scrape_urls(test_urls)
            
        print(f"✅ Successfully scraped {len(contents)} URLs")
        for content in contents:
            print(f"   - {content.url}: {content.word_count} words")
            
        return contents
        
    except Exception as e:
        print(f"❌ Scraper test failed: {e}")
        return []

def test_detector(contents):
    """Test the duplicate detector."""
    print("🧪 Testing Duplicate Detector...")
    
    if len(contents) < 2:
        print("⚠️  Need at least 2 content items for detection test")
        
        # Create test content
        test_contents = [
            type('Content', (), {
                'url': 'test1',
                'content': 'This is a test document about artificial intelligence and machine learning.',
                'title': 'Test 1',
                'word_count': 15
            })(),
            type('Content', (), {
                'url': 'test2',
                'content': 'This is another test document about AI and machine learning techniques.',
                'title': 'Test 2',
                'word_count': 15
            })()
        ]
        contents = test_contents
    
    try:
        config = Config()
        detector = DuplicateDetector(config)
        results = detector.detect_duplicates(contents)
        
        print(f"✅ Analyzed {len(contents)} content items")
        print(f"✅ Found {len(results)} potential duplicates")
        
        for result in results:
            if result.is_duplicate:
                print(f"   🔴 Duplicate: {result.url1} vs {result.url2} ({result.similarity_score:.2%})")
            else:
                print(f"   🟢 Similar: {result.url1} vs {result.url2} ({result.similarity_score:.2%})")
                
        return results
        
    except Exception as e:
        print(f"❌ Detector test failed: {e}")
        return []

async def test_analyzer():
    """Test the complete analyzer."""
    print("🧪 Testing Complete Analyzer...")
    
    test_urls = [
        "https://httpbin.org/html",
        "https://example.com",
    ]
    
    try:
        analyzer = DuplicateAnalyzer()
        results = await analyzer.analyze_urls(test_urls)
        
        if "error" in results:
            print(f"❌ Analyzer test failed: {results['error']}")
            return None
            
        summary = results["summary"]
        print(f"✅ Analysis complete!")
        print(f"   📊 Total URLs: {summary['total_urls']}")
        print(f"   📄 Content scraped: {summary['content_scraped']}")
        print(f"   🔍 Duplicates found: {summary['duplicates_found']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        return None

def test_config():
    """Test configuration loading."""
    print("🧪 Testing Configuration...")
    
    try:
        config = Config()
        config_dict = config.to_dict()
        
        print(f"✅ Configuration loaded with {len(config_dict)} settings")
        print(f"   🔧 Semantic threshold: {config.SEMANTIC_THRESHOLD}")
        print(f"   🔧 Max workers: {config.MAX_WORKERS}")
        print(f"   🔧 Min content length: {config.MIN_CONTENT_LENGTH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def run_all_tests():
    """Run all system tests."""
    print("🚀 Starting System Tests...\n")
    
    # Test configuration
    config_ok = test_config()
    
    # Test scraper
    contents = await test_scraper()
    
    # Test detector
    results = test_detector(contents)
    
    # Test analyzer
    analyzer_results = await test_analyzer()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 50)
    
    if config_ok:
        print("✅ Configuration: OK")
    else:
        print("❌ Configuration: FAILED")
        
    if contents:
        print("✅ Scraper: OK")
    else:
        print("❌ Scraper: FAILED")
        
    if results:
        print("✅ Detector: OK")
    else:
        print("❌ Detector: FAILED")
        
    if analyzer_results and "error" not in analyzer_results:
        print("✅ Analyzer: OK")
    else:
        print("❌ Analyzer: FAILED")
        
    print("\n🎉 System tests completed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
