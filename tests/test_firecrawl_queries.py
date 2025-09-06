#!/usr/bin/env python3
"""Extended Firecrawl search tests.

These tests are designed for manual review of output while remaining resilient
in environments where the self-hosted Firecrawl service isn't available.

Behavior:
    - If Firecrawl base url is missing, tests skip.
    - API key is optional for self-hosted usage; if present it's used.
  - If the service is unreachable or returns non-200, tests skip with reason.
  - Otherwise, they print formatted search results for multiple queries.
  - Integration test exercises SearchUtils legacy methods now backed by Firecrawl.
"""
from __future__ import annotations

import os
import sys
import json
import time
import textwrap
import logging
from typing import List, Dict, Any
from datetime import datetime

try:  # Optional dependency for test execution
    import pytest  # type: ignore
except Exception:  # pragma: no cover
    class _PytestShim:  # minimal skip shim
        @staticmethod
        def skip(msg):
            print(f"[pytest skip] {msg}")
        
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    if condition:
                        def wrapped(*args, **kwargs):
                            print(f"[pytest skipif] {reason}")
                            return
                        return wrapped
                    return func
                return decorator
                
    pytest = _PytestShim()  # type: ignore
import requests

# Allow importing project modules when running directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, 'tests', '_artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, 'firecrawl_results.txt')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from bot.helper_functions.config_manager import ConfigManager
    from bot.helper_functions.search_utils import SearchUtils
except Exception:
    ConfigManager = None  # type: ignore
    SearchUtils = None  # type: ignore


def _load_project_config_yaml() -> Dict[str, Any]:
    cfg_path = os.path.join(PROJECT_ROOT, 'config', 'config.yml')
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            print("PyYAML not available, falling back to environment variables")
    return {}


def _get_firecrawl_settings():
    cfg_yaml = _load_project_config_yaml()
    base = (cfg_yaml.get('firecrawl_base_url') or os.getenv('FIRECRAWL_BASE_URL') or '').strip()
    key = (cfg_yaml.get('firecrawl_api_key') or os.getenv('FIRECRAWL_API_KEY') or '').strip()
    
    # Check for placeholder values that indicate incomplete configuration
    if key in ('fc-YOUR_API_KEY', 'YOUR_API_KEY', 'CHANGE_ME'):
        key = ''
    
    print(
        f"Firecrawl config - base_url: {'<set>' if base else '<missing>'}, "
        f"api_key: {'<set>' if key else '<none/optional>'}"
    )
    return base, key


def _require_service_or_skip():
    base, key = _get_firecrawl_settings()
    if not base:
        pytest.skip('Firecrawl not configured (missing base url).')
    
    # Validate the base URL format
    if not base.startswith(('http://', 'https://')):
        base = f'http://{base}'
    
    url = base.rstrip('/') + '/v2/search'
    logging.getLogger(__name__).debug(f"Testing Firecrawl connectivity at: {url}")

    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'

    try:
        r = requests.post(url, json={'query': 'connectivity test', 'limit': 1}, timeout=10, headers=headers)
    except requests.ConnectionError as e:
        pytest.skip(f'Firecrawl service unreachable (connection error): {e}')
    except requests.Timeout as e:
        pytest.skip(f'Firecrawl service unreachable (timeout): {e}')
    except Exception as e:
        pytest.skip(f'Firecrawl service unreachable: {e}')
    
    if r.status_code != 200:
        logging.getLogger(__name__).debug(f"Response headers: {dict(r.headers)}")
        logging.getLogger(__name__).debug(f"Response body: {r.text[:500]}")
        pytest.skip(f'Firecrawl search endpoint returned HTTP {r.status_code}: {r.text[:200]}')
    
    return base, key


def _pretty_print_results(query: str, data: Dict[str, Any]):
    # Handle different possible response structures
    web_results = []
    if 'data' in data and isinstance(data['data'], dict) and 'web' in data['data']:
        web_results = data['data']['web']
    elif 'data' in data and isinstance(data['data'], list):
        web_results = data['data']
    elif 'web' in data:
        web_results = data['web']
    elif 'results' in data:
        web_results = data['results']
    
    logging.getLogger(__name__).debug(
        f"===== Firecrawl Results for: {query!r} (showing {len(web_results)} items) ====="
    )
    
    if not web_results:
        logging.getLogger(__name__).debug("No results found or unexpected response structure:")
        logging.getLogger(__name__).debug(
            json.dumps(data, indent=2)[:500] + ("..." if len(json.dumps(data)) > 500 else "")
        )
        return
    
    for i, item in enumerate(web_results, 1):
        title = item.get('title') or '<no title>'
        desc = (item.get('description') or item.get('snippet') or '').replace('\n', ' ').strip()
        url = item.get('url') or ''
        if len(desc) > 220:
            desc = desc[:217] + '...'
        logging.getLogger(__name__).debug(f"{i:02d}. {title}\n    {desc}\n    {url}")
    
    # Persist to artifact file for later manual inspection (append)
    try:
        with open(ARTIFACT_PATH, 'a', encoding='utf-8') as f:
            f.write(f"\n===== {datetime.utcnow().isoformat()}Z Query: {query!r} =====\n")
            for i, item in enumerate(web_results, 1):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.getLogger(__name__).debug(f"Failed writing artifact: {e}")


@pytest.mark.skipif(
    not os.getenv('FIRECRAWL_BASE_URL') and not _load_project_config_yaml().get('firecrawl_base_url'),
        reason="Firecrawl not configured (no base url)"
)
def test_firecrawl_multiple_queries():
    """Query Firecrawl with a variety of user-style prompts and display output.

    This does not assert semantic quality—only structural success.
    """
    base, key = _require_service_or_skip()
    url = base.rstrip('/') + '/v2/search'
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

    queries: List[str] = [
        'openai',
        'latest India-USA trade war developments',
        'latest China-Russia-India relations and trade dynamics',
        'global economic outlook 2025',
        'Indian economy growth projections for 2025 and beyond',
    ]

    for q in queries:
        payload = {'query': q, 'limit': 3, 'sources': ['news']}
        start = time.time()
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        elapsed = (time.time() - start) * 1000
        assert r.status_code == 200, f"HTTP {r.status_code} for query {q}: {r.text[:200]}"
        data = r.json()
        # Check if response has success field or data field
        if 'success' in data:
            assert data.get('success'), f"Firecrawl did not report success for query {q}: {data}"
        elif 'data' not in data:
            # If neither success nor data field, print the response for debugging
            print(f"Unexpected response structure for query {q}: {data}")
        _pretty_print_results(q, data)
        logging.getLogger(__name__).debug(f"Query time: {elapsed:.1f} ms")

    logging.getLogger(__name__).debug(f"Consolidated Firecrawl artifact written to: {ARTIFACT_PATH}")


@pytest.mark.skipif(
    not os.getenv('FIRECRAWL_BASE_URL') and not _load_project_config_yaml().get('firecrawl_base_url'),
        reason="Firecrawl not configured (no base url)"
)
def test_firecrawl_integration_search_utils():
    """Exercise SearchUtils legacy methods now backed by Firecrawl.

    If LlamaIndex isn't installed, skip gracefully.
    """
    base, key = _require_service_or_skip()
    if ConfigManager is None or SearchUtils is None:
        pytest.skip('SearchUtils/ConfigManager imports unavailable.')

    # Instantiate config manager / search utils
    cm = ConfigManager()
    su = SearchUtils(cm)

    # Quick sanity: ensure our config manager sees firecrawl values (not required for pass)
    logging.getLogger(__name__).debug(f"Firecrawl base (cm): {getattr(cm, 'firecrawl_base_url', None)}")

    # Use legacy method names
    try:
        answer = su.get_bing_results('openai')
        assert isinstance(answer, str)
        logging.getLogger(__name__).debug('===== get_bing_results("openai") Answer (truncated) =====')
        logging.getLogger(__name__).debug(textwrap.shorten(answer, width=500, placeholder='...'))
    except ImportError as e:
        pytest.skip(f'LlamaIndex dependency missing for simple_query: {e}')

    try:
        news = su.get_bing_news_results('openai company updates')
        assert isinstance(news, str)
        logging.getLogger(__name__).debug('===== get_bing_news_results("openai company updates") Summary (truncated) =====')
        logging.getLogger(__name__).debug(textwrap.shorten(news, width=500, placeholder='...'))
    except ImportError as e:
        pytest.skip(f'LlamaIndex dependency missing for summarize: {e}')

    # Check that the snippet file was written
    upload_dir = getattr(cm, 'UPLOAD_FOLDER', './data')
    snippet_path = os.path.join(upload_dir, 'bing_results.txt')
    assert os.path.exists(snippet_path), 'Expected snippet file not found.'
    size = os.path.getsize(snippet_path)
    logging.getLogger(__name__).debug(f"Snippet file: {snippet_path} ({size} bytes)")
    # Show first few lines for manual inspection
    try:
        with open(snippet_path, 'r', encoding='utf-8', errors='ignore') as f:
            head_lines = []
            for _ in range(5):
                try:
                    line = next(f)
                    head_lines.append(line)
                except StopIteration:
                    break
            head_lines = ''.join(head_lines)
        logging.getLogger(__name__).debug('===== Snippet File Head =====')
        logging.getLogger(__name__).debug(head_lines)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Could not read snippet file head: {e}")


def test_firecrawl_config_loading():
    """Test that Firecrawl configuration can be loaded correctly."""
    base, key = _get_firecrawl_settings()
    
    # This test should always pass - it just verifies config loading works
    assert isinstance(base, str), "Base URL should be a string"
    assert isinstance(key, str), "API key should be a string"
    
    logging.getLogger(__name__).debug("Configuration loaded successfully")
    logging.getLogger(__name__).debug(f"Base URL configured: {bool(base)}")
    logging.getLogger(__name__).debug(f"API Key configured (optional): {bool(key)}")
    if not base:
        logging.getLogger(__name__).debug("Firecrawl is not configured (missing base url).")
    else:
        logging.getLogger(__name__).debug("Firecrawl base url configured (API key optional).")


def _manual_run():  # pragma: no cover
    """Allow direct execution: python tests/test_firecrawl_queries.py
    Shows output regardless of pytest capture settings."""
    quiet = os.getenv('QUIET_MODE') or os.getenv('FIRECRAWL_QUIET')
    if not quiet:
        print("[Manual Firecrawl Demo]")
    base, key = _get_firecrawl_settings()
    if not base:
        if not quiet:
            print("Firecrawl not configured. Set firecrawl_base_url.")
        return
    # Run the multiple queries test logic
    try:
        test_firecrawl_multiple_queries()
    except SystemExit:
        pass
    if not quiet:
        print(f"Results artifact: {ARTIFACT_PATH}")


if __name__ == '__main__':  # pragma: no cover
    _manual_run()
