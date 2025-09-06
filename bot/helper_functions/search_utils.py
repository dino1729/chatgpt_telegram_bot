"""
Search and web scraping utilities
"""
import os
import logging
import requests
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

# Optional imports with defensive loading
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    Article = None
    NEWSPAPER_AVAILABLE = False
    logging.warning("Newspaper library not available")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup library not available")

try:
    from llama_index.agent.openai import OpenAIAgent
    from llama_index.tools.weather import OpenWeatherMapToolSpec
    from llama_index.tools.bing_search import BingSearchToolSpec
    from llama_index.core import Settings
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    OpenAIAgent = None
    OpenWeatherMapToolSpec = None
    BingSearchToolSpec = None
    Settings = None
    LLAMA_INDEX_AVAILABLE = False
    logging.warning("LlamaIndex tools not available")

from .file_utils import FileUtils

class SearchUtils:
    """Utilities for web search, news search and content extraction.

    Bing API has been deprecated in this project; these helpers now use Firecrawl
    (self-hosted) for web/news search while keeping the old public method names
    (get_bing_results, get_bing_news_results, get_bing_agent) for backward
    compatibility with existing code paths. Internally we call Firecrawl.
    """
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.file_utils = FileUtils(config_manager)
        
    def clear_all_files(self):
        """Clear all files from upload folder"""
        self.file_utils.clear_all_files()
                
    def text_extractor(self, url, debug=False):
        """
        Extract text content from a URL
        
        Args:
            url: URL to extract text from
            debug: Enable debug logging
            
        Returns:
            Extracted text content
        """
        if not url:
            return None
            
        # Extract the article
        article = Article(url)
        try:
            article.download()
            article.parse()
            # Check if the article text has at least 75 words
            if len(article.text.split()) < 75:
                raise Exception("Article is too short. Probably the article is behind a paywall.")
        except Exception as e:
            if debug:
                logging.warning(f"Failed to download and parse article from URL using newspaper package: {url}. Error: {str(e)}")
            # Try an alternate method using requests and beautifulsoup
            try:
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')
                article.text = soup.get_text()
            except Exception as e:
                if debug:
                    logging.warning(f"Failed to download article using beautifulsoup method from URL: {url}. Error: {str(e)}")
        return article.text

    def save_extracted_text_to_file(self, text, filename):
        """
        Save extracted text to a file
        
        Args:
            text: Text content to save
            filename: Name of the file to save to
            
        Returns:
            Success message with file path
        """
        return self.file_utils.save_text_to_file(text, filename)

    # ------------------------------------------------------------------
    # Firecrawl integration helpers
    # ------------------------------------------------------------------
    def _firecrawl_base_url(self) -> Optional[str]:
        return getattr(self.config, 'firecrawl_base_url', None) or os.getenv("FIRECRAWL_BASE_URL")

    def _firecrawl_api_key(self) -> Optional[str]:
        return getattr(self.config, 'firecrawl_api_key', None) or os.getenv("FIRECRAWL_API_KEY")

    def _firecrawl_search(
        self,
        query: str,
        sources: Optional[List[Union[str, Dict[str, Any]]]] = None,
        limit: int = 5,
        categories: Optional[List[Union[str, Dict[str, Any]]]] = None,
        tbs: Optional[str] = None,
        location: Optional[str] = None,
        timeout: int = 60000,
        ignore_invalid_urls: bool = False,
        scrape_options: Optional[Dict[str, Any]] = None,
        raw: bool = False,
    ) -> Dict[str, Any]:
        """Call the Firecrawl v2 /search endpoint with extended options from OpenAPI spec.

        This extends the original helper while keeping positional arguments (query, sources, limit)
        for backward compatibility with existing callers. Additional parameters are keyword-only.

        Args:
            query: Search query string.
            sources: Sources list (e.g. ["web", "news", "images"]). Can also supply dicts with {"type": ...}.
            limit: Maximum number of results to return (1-100).
            categories: Optional categories (e.g. ["github", "research"]). Same coercion rules as sources.
            tbs: Time-based search parameter (e.g. 'qdr:d').
            location: Location parameter for search results.
            timeout: Timeout in milliseconds for the server-side search (default 60000).
            ignore_invalid_urls: Exclude URLs invalid for follow-on Firecrawl endpoints.
            scrape_options: Dict of scrapeOptions per spec (formats, onlyMainContent, includeTags, etc.).
            raw: When True, return the full JSON response (including 'success', 'data', 'warning').

        Returns:
            dict: Either the full JSON (raw=True) or just the 'data' subsection (default) or empty dict on error.
        """
        base = self._firecrawl_base_url()
        key = self._firecrawl_api_key()
        if not base:
            logging.error("Firecrawl base url not configured (firecrawl_base_url)")
            return {}

        # Input sanitation / coercion
        limit = max(1, min(int(limit), 100)) if isinstance(limit, int) else 5

        def _coerce(list_or_none: Optional[List[Union[str, Dict[str, Any]]]]):
            if not list_or_none:
                return None
            coerced = []
            for item in list_or_none:
                if isinstance(item, str):
                    coerced.append(item)
                elif isinstance(item, dict):
                    # Only keep known keys to avoid accidental leakage of internal data
                    sub = {k: v for k, v in item.items() if k in {"type", "tbs", "location"}}
                    if sub:
                        coerced.append(sub)
                # ignore other types silently
            return coerced or None

        sources_payload = _coerce(sources)
        categories_payload = _coerce(categories)

        # Scrape options sanitation
        so = scrape_options or {}
        if so:
            # Provide defaults mirroring spec if user omitted typical keys
            if 'onlyMainContent' not in so:
                so['onlyMainContent'] = True
            if 'removeBase64Images' not in so:
                so['removeBase64Images'] = True
            if 'blockAds' not in so:
                so['blockAds'] = True
            # Formats: allow user to pass list of strings OR list of dicts
            fmts = so.get('formats')
            if fmts and isinstance(fmts, list):
                cleaned_fmts = []
                for f in fmts:
                    if isinstance(f, str):
                        cleaned_fmts.append(f)
                    elif isinstance(f, dict) and 'type' in f:
                        cleaned_fmts.append({k: v for k, v in f.items() if k in {'type', 'fullPage', 'quality', 'viewport', 'schema', 'prompt', 'modes', 'tag'}})
                if cleaned_fmts:
                    so['formats'] = cleaned_fmts
        
        payload: Dict[str, Any] = {
            'query': query,
            'limit': limit,
        }
        if sources_payload:
            payload['sources'] = sources_payload
        if categories_payload:
            payload['categories'] = categories_payload
        if tbs:
            payload['tbs'] = tbs
        if location:
            payload['location'] = location
        if timeout:
            payload['timeout'] = int(timeout)
        if ignore_invalid_urls:
            payload['ignoreInvalidURLs'] = True
        if so:
            payload['scrapeOptions'] = so

        url = base.rstrip('/') + '/v2/search'
        headers = {'Content-Type': 'application/json'}
        if key:  # Authorization optional for self-hosted deployments
            headers['Authorization'] = f'Bearer {key}'
        # Convert ms timeout to seconds for requests (add small cushion but cap)
        req_timeout_seconds = min(max(int(timeout) / 1000.0 + 5, 10), 120) if timeout else 30

        try:
            resp = requests.post(url, json=payload, timeout=req_timeout_seconds)
        except requests.RequestException as e:
            logging.error(f"Firecrawl request error: {e}")
            return {}

        if resp.status_code != 200:
            logging.error(f"Firecrawl API error {resp.status_code}: {resp.text[:500]}")
            return {}

        try:
            json_resp = resp.json()
        except Exception as e:
            logging.error(f"Failed parsing Firecrawl response: {e}")
            return {}

        if raw:
            return json_resp
        return json_resp.get('data', {}) if isinstance(json_resp, dict) else {}

    def _format_firecrawl_web_results(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        web = data.get('web') or []
        formatted = []
        for item in web:
            formatted.append({
                'title': item.get('title') or '',
                'snippet': item.get('description') or '',
                'url': item.get('url') or ''
            })
        return formatted

    def _format_firecrawl_news_results(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        news = data.get('news') or []
        formatted = []
        for item in news:
            formatted.append({
                'title': item.get('title') or '',
                'snippet': item.get('snippet') or item.get('description') or '',
                'url': item.get('url') or ''
            })
        return formatted

    def _format_firecrawl_image_results(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        images = data.get('images') or []
        formatted = []
        for item in images:
            formatted.append({
                'title': item.get('title') or '',
                'snippet': f"{item.get('imageWidth', '')}x{item.get('imageHeight', '')}",
                'url': item.get('imageUrl') or item.get('url') or ''
            })
        return formatted

    def _persist_search_snippets(
        self,
        query: str,
        results: List[Dict[str, str]],
        filename: str = "bing_results.txt",
        source_label: str = "web"
    ) -> None:
        """Persist snippets from search results to a file for downstream indexing.

        Appends (rather than overwrites) if the file already exists when multiple source
        types are being aggregated (e.g. web + news).
        """
        snippets: List[str] = []
        for r in results:
            if r.get('title'):
                snippets.append(r['title'])
            if r.get('snippet'):
                snippets.append(r['snippet'])
            if r.get('url'):
                snippets.append(r['url'])
        if not snippets:
            return
        body = '\n'.join(snippets)
        header = (
            f"\n[Firecrawl {source_label} results for '{query}' at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
        )
        output = header + body + '\n'
        # Append to maintain multi-source context
        mode = 'a' if os.path.exists(os.path.join(self.config.UPLOAD_FOLDER, filename)) else 'w'
        path = os.path.join(self.config.UPLOAD_FOLDER, filename)
        try:
            with open(path, mode, encoding='utf-8') as f:
                f.write(output)
        except Exception as e:
            logging.error(f"Failed writing snippet file {path}: {e}")

    # ------------------------------------------------------------------
    # Public (legacy) APIs renamed to Firecrawl
    # ------------------------------------------------------------------
    def get_bing_results(self, query, num=10):
        """Legacy method name retained. Performs a Firecrawl web/news blended search.

        Returns a synthesized answer using llama-index over collected snippets.
        """
        from .index_utils import IndexUtils
        self.clear_all_files()
        data = self._firecrawl_search(query, sources=["web"], limit=num)
        if not data:
            return ("I'm unable to search the internet because Firecrawl is not configured "
                    "or returned an error. Please try a regular chat instead.")
        results_web = self._format_firecrawl_web_results(data)
        self._persist_search_snippets(query, results_web, source_label='web')
        # If news also present implicitly, capture
        if 'news' in data:
            results_news = self._format_firecrawl_news_results(data)
            self._persist_search_snippets(query, results_news, source_label='news')
        index_utils = IndexUtils(self.config)
        answer = str(index_utils.simple_query(self.config.UPLOAD_FOLDER, query)).strip()
        return answer or "No relevant information was found for your query."

    def get_bing_news_results(self, query, num=5):
        """Legacy news method name -> Firecrawl 'news' source search.

        Returns a summarized result using llama-index summarization.
        """
        from .index_utils import IndexUtils
        self.clear_all_files()
        data = self._firecrawl_search(query, sources=["news"], limit=num)
        if not data:
            return ("I'm unable to search for news because Firecrawl is not configured "
                    "or returned an error. Please try a regular chat instead.")
        # Prefer news formatting, fall back to web if API returns only web results
        if 'news' in data:
            results_news = self._format_firecrawl_news_results(data)
            self._persist_search_snippets(query, results_news, source_label='news')
        results_web = self._format_firecrawl_web_results(data)
        if results_web:
            self._persist_search_snippets(query, results_web, source_label='web')
        index_utils = IndexUtils(self.config)
        summary = str(index_utils.summarize(self.config.UPLOAD_FOLDER)).strip()
        return summary or "No recent news articles were found for your query."

    # ------------------------------------------------------------------
    # New public advanced search API
    # ------------------------------------------------------------------
    def firecrawl_search(
        self,
        query: str,
        *,
        limit: int = 5,
        sources: Optional[List[Union[str, Dict[str, Any]]]] = None,
        categories: Optional[List[Union[str, Dict[str, Any]]]] = None,
        tbs: Optional[str] = None,
        location: Optional[str] = None,
        timeout: int = 60000,
        ignore_invalid_urls: bool = False,
        scrape_options: Optional[Dict[str, Any]] = None,
        include_images: bool = False,
        include_news: bool = False,
        persist: bool = False,
    ) -> Dict[str, Any]:
        """Perform an advanced Firecrawl search and optionally persist snippets.

        Args mirror the internal helper plus convenience flags for selecting sources.
        """
        # Derive sources if not explicitly provided
        auto_sources: List[Union[str, Dict[str, Any]]] = sources or ["web"]
        if include_news and 'news' not in auto_sources:
            auto_sources.append('news')
        if include_images and 'images' not in auto_sources:
            auto_sources.append('images')

        data_section = self._firecrawl_search(
            query=query,
            sources=auto_sources,
            limit=limit,
            categories=categories,
            tbs=tbs,
            location=location,
            timeout=timeout,
            ignore_invalid_urls=ignore_invalid_urls,
            scrape_options=scrape_options,
            raw=False,
        )
        if not data_section:
            return {}

        if persist:
            if 'web' in data_section:
                self._persist_search_snippets(query, self._format_firecrawl_web_results(data_section), source_label='web')
            if 'news' in data_section:
                self._persist_search_snippets(query, self._format_firecrawl_news_results(data_section), source_label='news')
            if 'images' in data_section:
                self._persist_search_snippets(query, self._format_firecrawl_image_results(data_section), source_label='images')
        return data_section
    
    def get_bing_agent(self, query):
        """Legacy agent method. For now just proxy to get_bing_results.

        Could be enhanced to implement a multi-step tool use agent over Firecrawl
        but we keep it simple for backward compatibility.
        """
        return self.get_bing_results(query)
    
    def get_weather_data(self, query):
        """
        Get weather data using OpenWeatherMap
        
        Args:
            query: Weather query
            
        Returns:
            Weather information
        """
        weather_tool = OpenWeatherMapToolSpec(key=self.config.openweather_api_key)
        agent = OpenAIAgent.from_tools(
            weather_tool.to_tool_list(),
            llm=Settings.llm,
            verbose=False,
        )
        return str(agent.chat(query))
