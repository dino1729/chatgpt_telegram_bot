#!/usr/bin/env python3
"""Tests for advanced Firecrawl search parameters.

These tests are optional and will skip gracefully if Firecrawl is not configured.
They exercise the new SearchUtils.firecrawl_search API with extended parameters
(tbs, categories, scrapeOptions, multiple sources).
"""
from __future__ import annotations
import os
import sys
import logging
import argparse
import pytest

# Ensure project root is on sys.path when running directly (python3 tests/..)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_base_url_from_config() -> str:
    """Attempt to read firecrawl_base_url from config/config.yml (optional)."""
    cfg_path = os.path.join(PROJECT_ROOT, 'config', 'config.yml')
    if not os.path.exists(cfg_path):
        return ''
    try:  # pragma: no cover (best-effort)
        import yaml  # type: ignore
        with open(cfg_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return (data.get('firecrawl_base_url') or '').strip()
    except Exception:
        return ''

try:
    from bot.helper_functions.config_manager import ConfigManager
    from bot.helper_functions.search_utils import SearchUtils
except Exception:  # pragma: no cover - import errors cause skip
    ConfigManager = None  # type: ignore
    SearchUtils = None  # type: ignore


_CONFIG_BASE = _load_base_url_from_config()

@pytest.mark.skipif(
    not (os.getenv('FIRECRAWL_BASE_URL') or _CONFIG_BASE),
    reason='Firecrawl base URL not configured (env FIRECRAWL_BASE_URL or config.yml)'
)
def test_advanced_firecrawl_search_basic():
    if ConfigManager is None or SearchUtils is None:
        pytest.skip('Dependencies unavailable')
    cm = ConfigManager()
    su = SearchUtils(cm)

    data = su.firecrawl_search(
        'open source vector databases',
        limit=3,
        include_news=True,
        include_images=False,
        tbs='qdr:w',  # last week
        scrape_options={
            'formats': ['markdown'],
            'onlyMainContent': True,
        },
        persist=False,
    )
    # Not asserting structure strongly; just ensure dictionary if service responded
    assert isinstance(data, dict)
    # If web present, ensure <= 3 results (soft check)
    if 'web' in data:
        assert len(data['web']) <= 3


@pytest.mark.skipif(
    not (os.getenv('FIRECRAWL_BASE_URL') or _CONFIG_BASE),
    reason='Firecrawl base URL not configured (env FIRECRAWL_BASE_URL or config.yml)'
)
def test_advanced_firecrawl_persist(tmp_path):
    if ConfigManager is None or SearchUtils is None:
        pytest.skip('Dependencies unavailable')
    cm = ConfigManager()
    # Override upload folder to temp to avoid polluting project data
    cm.UPLOAD_FOLDER = str(tmp_path)
    su = SearchUtils(cm)

    data = su.firecrawl_search(
        'latest machine learning research trends 2025',
        limit=2,
        sources=['web', 'news'],
        categories=['research'],
        include_images=False,
        tbs='qdr:m',  # last month
        scrape_options={
            'formats': ['markdown', {'type': 'rawHtml'}],
            'onlyMainContent': True,
            'blockAds': True,
        },
        persist=True,
    )
    assert isinstance(data, dict)
    snippet_path = os.path.join(cm.UPLOAD_FOLDER, 'bing_results.txt')
    # File may be empty if no results but should exist when persist=True & any data present
    if data:
        assert os.path.exists(snippet_path)
        # Basic sanity of content
        with open(snippet_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(2000)
        assert 'Firecrawl' in content or 'results' in content


@pytest.mark.skipif(
    not (os.getenv('FIRECRAWL_BASE_URL') or _CONFIG_BASE),
    reason='Firecrawl base URL not configured (env FIRECRAWL_BASE_URL or config.yml)'
)
def test_coherent_answer_generation():
    """Ensure get_bing_results returns a synthesized narrative answer rather than raw links.

    Heuristic checks:
      - Answer is a non-empty string
      - Contains at least one space (more than a single token)
      - Does NOT look like a simple list of URLs (e.g., does not start with 'http')
      - Contains fewer raw 'http' substrings than 50% of number of words (avoids pure link dumps)
    Skips if LlamaIndex dependencies are unavailable.
    """
    if ConfigManager is None or SearchUtils is None:
        pytest.skip('Dependencies unavailable')
    cm = ConfigManager()
    su = SearchUtils(cm)

    # Use a query that should yield explanatory content (pytest path)
    answer = su.get_bing_results('What are key benefits of vector databases for AI retrieval?')
    assert isinstance(answer, str) and answer.strip(), 'Expected non-empty answer string'
    text = answer.strip()
    assert ' ' in text, 'Answer appears too short to be coherent'
    assert not text.lower().startswith('http'), 'Answer should not start with a URL'
    word_count = len(text.split())
    http_count = text.lower().count('http')
    # Allow some links but ensure not dominated by them
    assert http_count <= max(1, word_count // 4), 'Answer contains too many raw URLs relative to length'

    # Basic semantic heuristic: presence of explanatory terms
    explanatory_tokens = {'benefit', 'advantages', 'improve', 'performance', 'search', 'retrieval'}
    if word_count > 15:  # Only check if answer has substance
        overlap = explanatory_tokens.intersection({w.strip('.,:;!?').lower() for w in text.split()})
        assert overlap, 'Answer lacks expected explanatory vocabulary'


def _resolve_base_url() -> str:
    base = os.getenv('FIRECRAWL_BASE_URL') or _CONFIG_BASE
    if base:
        return base
    if ConfigManager is not None:
        try:
            cm_tmp = ConfigManager()
            return getattr(cm_tmp, 'firecrawl_base_url', '') or ''
        except Exception:  # pragma: no cover
            return ''
    return ''


def _coherent_answer(query: str) -> str:
    if ConfigManager is None or SearchUtils is None:
        raise RuntimeError('Dependencies unavailable (ConfigManager/SearchUtils)')
    cm = ConfigManager()
    su = SearchUtils(cm)
    return su.get_bing_results(query)


if __name__ == '__main__':  # CLI mode: only print coherent answer
    parser = argparse.ArgumentParser(description='Fetch a coherent Firecrawl-backed answer.')
    parser.add_argument('query', nargs='*', help='User query to answer (provide as plain text).')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging / extra diagnostics.')
    parser.add_argument('--validate', action='store_true', help='Run heuristic validation checks (like pytest test).')
    args = parser.parse_args()
    
    def _configure_logging(debug: bool):
        # Reset existing handlers installed elsewhere (e.g., config_manager)
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        level = logging.DEBUG if debug else logging.ERROR
        logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
        # Silence noisy third-party loggers when not debug
        if not debug:
            noisy = [
                'urllib3', 'requests', 'httpx', 'openai', 'azure',
                'llama_index', 'llama_index.core', 'llama_index.readers',
                'llama_index.token_counter', 'bot', '__main__'
            ]
            for name in noisy:
                logging.getLogger(name).setLevel(logging.CRITICAL)
            # Additionally suppress HTTPConnection debug if previously enabled
            try:
                import http.client as http_client  # type: ignore
                http_client.HTTPConnection.debuglevel = 0
            except Exception:
                pass
        else:
            # In debug keep llama_index at INFO to avoid excessive internals
            logging.getLogger('llama_index').setLevel(logging.INFO)
    
    _configure_logging(args.debug)
    # Hint to llama index via env var
    if not args.debug:
        os.environ['LLAMA_INDEX_LOG_LEVEL'] = 'ERROR'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    user_query = ' '.join(args.query).strip() or 'What are key benefits of vector databases for AI retrieval?'
    base_url = _resolve_base_url()
    if not base_url:
        print('Error: Firecrawl base URL not configured (env FIRECRAWL_BASE_URL or config.yml).', file=sys.stderr)
        sys.exit(1)

    try:
        answer = _coherent_answer(user_query)
    except Exception as e:  # pragma: no cover
        if args.debug:
            logging.exception('Failed generating answer')
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(2)

    # Optional heuristic validation (only if requested)
    if args.validate:
        txt = answer.strip()
        if not txt or txt.lower().startswith('http'):
            print('Warning: answer may not be coherent (starts with URL or empty).', file=sys.stderr)
        if txt:
            wc = len(txt.split())
            http_c = txt.lower().count('http')
            if http_c > max(1, wc // 4):
                print('Warning: answer contains many URLs relative to length.', file=sys.stderr)

    # Print ONLY the coherent answer (no extra formatting) to stdout
    print(answer.strip())
