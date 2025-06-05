"""
Search and web scraping utilities
"""
import os
import logging
import requests
from datetime import datetime

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
    """Utilities for web search and content extraction"""
    
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

    def get_bing_results(self, query, num=10):
        """
        Get Bing search results
        
        Args:
            query: Search query
            num: Number of results to fetch
            
        Returns:
            Search results summary
        """
        # Validate Bing API key
        if not self.config.bing_api_key or self.config.bing_api_key.strip() == "":
            logging.error("Bing API key is not configured or empty")
            return "I'm unable to search the internet because the search service is not properly configured. Please try a regular chat instead."
        
        from .index_utils import IndexUtils
        
        self.clear_all_files()
        
        # Construct a request
        mkt = 'en-US'
        params = {'q': query, 'mkt': mkt, 'count': num, 'responseFilter': ['Webpages', 'News']}
        headers = {'Ocp-Apim-Subscription-Key': self.config.bing_api_key}
        response = requests.get(self.config.bing_endpoint, headers=headers, params=params)
        
        # Check for API errors
        if response.status_code != 200:
            logging.error(f"Bing API error: {response.status_code} - {response.text}")
            return f"I'm unable to search the internet right now due to a search service error (HTTP {response.status_code}). Please try a regular chat instead."
        
        response_data = response.json()
        
        # Check if the response contains the expected webPages field
        if 'webPages' not in response_data or 'value' not in response_data['webPages']:
            logging.error(f"Unexpected Bing API response structure: {response_data}")
            return "I'm unable to search the internet right now due to an unexpected search service response. Please try a regular chat instead."

        # Extract snippets and append them into a single text variable
        all_snippets = [result['snippet'] for result in response_data['webPages']['value']]
        combined_snippets = '\n'.join(all_snippets)
        
        # Format the results as a string
        output = f"Here is the context from Bing for the query: '{query}'. Current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n"
        output += combined_snippets

        # Save the output to a file
        self.save_extracted_text_to_file(output, "bing_results.txt")
        
        # Query the results using llama-index
        index_utils = IndexUtils(self.config)
        answer = str(index_utils.simple_query(self.config.UPLOAD_FOLDER, query)).strip()
        return answer

    def get_bing_news_results(self, query, num=5):
        """
        Get Bing news search results
        
        Args:
            query: Search query
            num: Number of results to fetch
            
        Returns:
            News results summary
        """
        # Validate Bing API key
        if not self.config.bing_api_key or self.config.bing_api_key.strip() == "":
            logging.error("Bing API key is not configured or empty")
            return "I'm unable to search for news because the search service is not properly configured. Please try a regular chat instead."
        
        from .index_utils import IndexUtils
        
        self.clear_all_files()
        
        # Construct a request
        mkt = 'en-US'
        params = {'q': query, 'mkt': mkt, 'freshness': 'Day', 'count': num}
        headers = {'Ocp-Apim-Subscription-Key': self.config.bing_api_key}
        response = requests.get(self.config.bing_news_endpoint, headers=headers, params=params)
        
        # Check for API errors
        if response.status_code != 200:
            logging.error(f"Bing News API error: {response.status_code} - {response.text}")
            return f"I'm unable to search for news right now due to a search service error (HTTP {response.status_code}). Please try a regular chat instead."
        
        response_data = response.json()
        
        # Check if the response contains the expected value field
        if 'value' not in response_data:
            logging.error(f"Unexpected Bing News API response structure: {response_data}")
            return "I'm unable to search for news right now due to an unexpected search service response. Please try a regular chat instead."

        # Extract text from the urls and append them into a single text variable
        all_urls = [result['url'] for result in response_data['value']]
        all_snippets = [self.text_extractor(url) for url in all_urls]

        # Combine snippets with titles and article names
        combined_output = ""
        for i, (snippet, result) in enumerate(zip(all_snippets, response_data['value'])):
            title = f"Article {i + 1}: {result['name']}"
            if snippet and len(snippet.split()) >= 75:  # Check if article has at least 75 words
                combined_output += f"\n{title}\n{snippet}\n"

        # Format the results as a string
        output = f"Here's the scraped text from top {num} articles for the query: '{query}'. Current date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n"
        output += combined_output

        # Save the output to a file
        self.save_extracted_text_to_file(output, "bing_results.txt")
        
        # Summarize the bing search response
        index_utils = IndexUtils(self.config)
        bing_summary = str(index_utils.summarize(self.config.UPLOAD_FOLDER)).strip()
        return bing_summary
    
    def get_bing_agent(self, query):
        """
        Use Bing search agent for queries
        
        Args:
            query: Search query
            
        Returns:
            Agent response
        """
        bing_tool = BingSearchToolSpec(api_key=self.config.bing_api_key)
        agent = OpenAIAgent.from_tools(
            bing_tool.to_tool_list(),
            llm=Settings.llm,
            verbose=False,
        )
        return str(agent.chat(query))
    
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
