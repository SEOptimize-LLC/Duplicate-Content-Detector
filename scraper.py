"""Advanced web scraping module with intelligent content extraction."""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
import chardet
import time
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from config import Config

@dataclass
class ScrapedContent:
    """Data class for scraped content."""
    url: str
    title: str
    content: str
    meta_description: str
    headings: List[str]
    word_count: int
    language: str
    publish_date: Optional[str] = None
    author: Optional[str] = None
    images: List[str] = None
    links: List[str] = None
    content_hash: str = ""
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.links is None:
            self.links = []
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.content_hash:
            self.content_hash = hashlib.md5(
                self.content.encode('utf-8')
            ).hexdigest()

class WebScraper:
    """Advanced web scraper with caching and intelligent extraction."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.cache_dir = Path(config.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.config.REQUEST_TIMEOUT)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': self.config.USER_AGENT}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path for URL."""
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{url_hash}.json"
        
    def _load_from_cache(self, url: str) -> Optional[ScrapedContent]:
        """Load content from cache if available and valid."""
        if not self.config.CACHE_ENABLED:
            return None
            
        cache_path = self._get_cache_path(url)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check cache validity
            if time.time() - data.get('timestamp', 0) > self.config.CACHE_TTL:
                cache_path.unlink()
                return None
                
            return ScrapedContent(**data)
        except Exception as e:
            self.logger.warning(f"Cache load error for {url}: {e}")
            return None
            
    def _save_to_cache(self, content: ScrapedContent) -> None:
        """Save content to cache."""
        if not self.config.CACHE_ENABLED:
            return
            
        cache_path = self._get_cache_path(content.url)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(content.__dict__, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Cache save error for {content.url}: {e}")
            
    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content with retries and error handling."""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Detect encoding
                        encoding = chardet.detect(content)['encoding'] or 'utf-8'
                        return content.decode(encoding, errors='ignore')
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {e}")
                
            if attempt < self.config.MAX_RETRIES - 1:
                await asyncio.sleep(self.config.RATE_LIMIT_DELAY * (attempt + 1))
                
        return None
        
    def _remove_boilerplate(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove common boilerplate elements like nav, footer, sidebar."""
        # Remove navigation elements
        for element in soup.find_all(['nav', 'header', 'footer', 'aside']):
            element.decompose()
            
        # Remove common navigation classes
        nav_classes = [
            'nav', 'navbar', 'navigation', 'menu', 'main-menu',
            'primary-menu', 'secondary-menu', 'sidebar', 'widget',
            'footer', 'site-footer', 'page-footer', 'colophon',
            'social-links', 'social-media', 'share-buttons'
        ]
        
        for class_name in nav_classes:
            for element in soup.find_all(class_=lambda x: x and class_name in str(x).lower()):
                element.decompose()
                
        # Remove common IDs
        nav_ids = [
            'nav', 'navigation', 'menu', 'sidebar', 'footer',
            'header', 'masthead', 'colophon', 'social'
        ]
        
        for id_name in nav_ids:
            for element in soup.find_all(id=lambda x: x and id_name in str(x).lower()):
                element.decompose()
                
        return soup
        
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content, excluding boilerplate."""
        # Remove boilerplate first
        soup = self._remove_boilerplate(soup)
        
        # Common main content selectors
        content_selectors = [
            'main', 'article', '[role="main"]', '.content', '.main-content',
            '.post-content', '.entry-content', '#content', '#main-content',
            '.post-body', '.article-content', '.entry-summary', '.post-text'
        ]
        
        # Try to find main content area
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(separator=' ', strip=True)
                if len(text) > len(content):
                    content = text
                    
        if not content:
            # Fallback to body content, but remove scripts/styles
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            content = soup.get_text(separator=' ', strip=True)
            
        return content
        
    def _extract_with_newspaper(self, url: str, html: str) -> Optional[ScrapedContent]:
        """Extract content using newspaper3k for better accuracy."""
        try:
            article = Article(url)
            article.set_html(html)
            article.parse()
            
            if not article.text or len(article.text) < self.config.MIN_CONTENT_LENGTH:
                return None
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Clean the soup to remove boilerplate
            soup = self._remove_boilerplate(soup)
            
            # Extract headings from cleaned content
            headings = []
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                headings.extend([h.get_text(strip=True) for h in soup.find_all(tag)])
                
            # Extract images and links from cleaned content
            images = [img.get('src') for img in soup.find_all('img') if img.get('src')]
            links = [a.get('href') for a in soup.find_all('a') if a.get('href')]
            links = [urljoin(url, link) for link in links]
            
            return ScrapedContent(
                url=url,
                title=article.title or "",
                content=article.text,
                meta_description=article.meta_description or "",
                headings=headings,
                word_count=len(article.text.split()),
                language=article.meta_lang or "en",
                publish_date=article.publish_date.isoformat() if article.publish_date else None,
                author=", ".join(article.authors) if article.authors else None,
                images=images,
                links=links
            )
            
        except Exception as e:
            self.logger.error(f"Newspaper extraction error for {url}: {e}")
            return None
            
    def _extract_with_bs4(self, url: str, html: str) -> Optional[ScrapedContent]:
        """Fallback extraction using BeautifulSoup with boilerplate removal."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove boilerplate elements
            soup = self._remove_boilerplate(soup)
            
            # Remove remaining script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Extract title
            title = soup.find('title')
            title = title.get_text(strip=True) if title else ""
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_desc.get('content', '') if meta_desc else ""
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            if len(content) < self.config.MIN_CONTENT_LENGTH:
                return None
                
            # Extract headings from cleaned content
            headings = []
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                headings.extend([h.get_text(strip=True) for h in soup.find_all(tag)])
                
            # Extract images and links from cleaned content
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
            )
            
        except Exception as e:
            self.logger.error(f"BS4 extraction error for {url}: {e}")
            return None
            
    def _clean_content(self, text: str) -> str:
        """Clean extracted content by removing common patterns."""
        # Remove common navigation phrases
        nav_phrases = [
            'home', 'about', 'contact', 'privacy policy', 'terms of service',
            'copyright', 'all rights reserved', 'sitemap', 'search',
            'menu', 'navigation', 'footer', 'sidebar', 'widget'
        ]
        
        # Split into lines and filter
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 20:  # Skip very short lines
                # Skip lines with navigation phrases
                lower_line = line.lower()
                if not any(phrase in lower_line for phrase in nav_phrases):
                    cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
        
    async def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """Scrape a single URL with caching."""
        # Check cache first
        cached = self._load_from_cache(url)
        if cached:
            return cached
            
        html = await self._fetch_url(url)
        if not html:
            return None
            
        # Try newspaper extraction first
        content = self._extract_with_newspaper(url, html)
        if not content:
            # Fallback to BS4
            content = self._extract_with_bs4(url, html)
            
        if content:
            # Clean the content
            content.content = self._clean_content(content.content)
            content.word_count = len(content.content.split())
            
            # Skip if content is too short after cleaning
            if len(content.content) < self.config.MIN_CONTENT_LENGTH:
                return None
                
            self._save_to_cache(content)
            
        return content
        
    async def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """Scrape multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(self.config.MAX_WORKERS)
        
        async def scrape_with_semaphore(url: str) -> Optional[ScrapedContent]:
            async with semaphore:
                await asyncio.sleep(self.config.RATE_LIMIT_DELAY)
                return await self.scrape_url(url)
                
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and None results
        contents = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Scraping error: {result}")
            elif result:
                contents.append(result)
                
        return contents
        
    def extract_sitemap_urls(self, sitemap_url: str) -> List[str]:
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
                    nested_urls = self.extract_sitemap_urls(loc.text.strip())
                    urls.extend(nested_urls)
                    
            return urls
            
        except Exception as e:
            self.logger.error(f"Error extracting sitemap URLs: {e}")
            return []
