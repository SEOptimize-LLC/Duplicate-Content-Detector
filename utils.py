"""Utility functions for the duplicate content detector."""

import asyncio
import logging
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from config import Config
from scraper import WebScraper, ScrapedContent
from detector import DuplicateDetector, DuplicateResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DuplicateAnalyzer:
    """Command-line interface for duplicate content analysis."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.detector = DuplicateDetector(self.config)
        
    async def analyze_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Analyze a list of URLs for duplicate content."""
        logger.info(f"Starting analysis of {len(urls)} URLs")
        
        # Scrape content
        async with WebScraper(self.config) as scraper:
            contents = await scraper.scrape_urls(urls)
            
        if not contents:
            return {"error": "No content could be scraped"}
            
        # Detect duplicates
        results = self.detector.detect_duplicates(contents)
        
        # Prepare summary
        summary = {
            "total_urls": len(urls),
            "content_scraped": len(contents),
            "duplicates_found": sum(1 for r in results if r.is_duplicate),
            "analysis_date": datetime.now().isoformat(),
            "config": self.config.to_dict()
        }
        
        return {
            "summary": summary,
            "results": results,
            "contents": contents
        }
        
    def analyze_from_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze URLs from a file."""
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
                
            return asyncio.run(self.analyze_urls(urls))
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"error": str(e)}
            
    def analyze_sitemap(self, sitemap_url: str) -> Dict[str, Any]:
        """Analyze all URLs from a sitemap."""
        try:
            scraper = WebScraper(self.config)
            urls = scraper.extract_sitemap_urls(sitemap_url)
            
            if not urls:
                return {"error": "No URLs found in sitemap"}
                
            return asyncio.run(self.analyze_urls(urls))
            
        except Exception as e:
            logger.error(f"Error processing sitemap {sitemap_url}: {e}")
            return {"error": str(e)}
            
    def export_results(self, results: Dict[str, Any], output_dir: str = "./results"):
        """Export analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export summary
        summary_path = output_path / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(results["summary"], f, indent=2, default=str)
            
        # Export detailed results
        if "results" in results:
            results_path = output_path / f"detailed_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                # Convert DuplicateResult objects to dicts
                results_data = [r.__dict__ for r in results["results"]]
                json.dump(results_data, f, indent=2, default=str)
                
        # Export CSV
        if "results" in results:
            csv_path = output_path / f"results_{timestamp}.csv"
            self._export_csv(results["results"], csv_path)
            
        logger.info(f"Results exported to {output_path}")
        
    def _export_csv(self, results: List[DuplicateResult], file_path: Path):
        """Export results to CSV format."""
        data = []
        for result in results:
            row = {
                'url1': result.url1,
                'url2': result.url2,
                'similarity_score': result.similarity_score,
                'confidence': result.confidence,
                'is_duplicate': result.is_duplicate,
                'common_content': result.common_content,
                'differences': result.differences
            }
            
            # Add individual scores
            for method, score in result.metadata.get('individual_scores', {}).items():
                row[f'score_{method}'] = score
                
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable report."""
        if "error" in results:
            return f"Error: {results['error']}"
            
        summary = results["summary"]
        report = f"""
# Duplicate Content Analysis Report

Generated: {summary['analysis_date']}

## Summary
- **Total URLs Analyzed**: {summary['total_urls']}
- **Content Successfully Scraped**: {summary['content_scraped']}
- **Duplicate Pairs Found**: {summary['duplicates_found']}

## Configuration
- Semantic Threshold: {summary['config']['SEMANTIC_THRESHOLD']}
- Minimum Content Length: {summary['config']['MIN_CONTENT_LENGTH']}
- Max Workers: {summary['config']['MAX_WORKERS']}

## Duplicate Pairs
"""
        
        if "results" in results:
            for i, result in enumerate(results["results"]):
                if result.is_duplicate:
                    report += f"\n{i+1}. **{result.url1}** vs **{result.url2}**"
                    report += f"\n   Similarity: {result.similarity_score:.2%}"
                    report += f"\n   Confidence: {result.confidence:.2%}\n"
                    
        return report


def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Powered Duplicate Content Detector")
    parser.add_argument("--urls", nargs="+", help="List of URLs to analyze")
    parser.add_argument("--file", help="File containing URLs (one per line)")
    parser.add_argument("--sitemap", help="Sitemap URL to analyze")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--config", help="JSON config file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = Config()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
                config.update_from_dict(config_dict)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            
    analyzer = DuplicateAnalyzer(config)
    
    # Run analysis
    if args.urls:
        results = asyncio.run(analyzer.analyze_urls(args.urls))
    elif args.file:
        results = analyzer.analyze_from_file(args.file)
    elif args.sitemap:
        results = analyzer.analyze_sitemap(args.sitemap)
    else:
        print("Please provide URLs, file, or sitemap")
        return
        
    # Export results
    analyzer.export_results(results, args.output)
    
    # Print summary
    if "summary" in results:
        summary = results["summary"]
        print(f"\nAnalysis Complete!")
        print(f"Total URLs: {summary['total_urls']}")
        print(f"Content Scraped: {summary['content_scraped']}")
        print(f"Duplicates Found: {summary['duplicates_found']}")
        print(f"Results exported to: {args.output}")


if __name__ == "__main__":
    main()
