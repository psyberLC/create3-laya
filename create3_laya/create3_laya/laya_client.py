"""Minimal client for a local Laya server (`laya-serve`), stdlib only.

Laya is a non-generative decision model: each request sends a `state` plus typed questions
and gets back a probability for every option.

  POST {url}/v1/systemone
  {"state": ..., "model": "english", "questions": {key: {type, instructions, criteria}}}
  -> {"model": ..., "answers": {key: {"choice", "probabilities", ...}}, "usage": {...}}

Start the server with scripts/serve_laya.ps1 (or .sh); it listens on 127.0.0.1:8000.
"""

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_URL = 'http://127.0.0.1:8000'
URL_ENV = 'LAYA_URL'
MODEL_ENV = 'LAYA_MODEL'
API_KEY_ENV = 'LAYA_API_KEY'
RETRYABLE_STATUSES = (429, 503, 529)


class LayaError(RuntimeError):
    """A failed call. `retryable` errors are transient; others mean stop and fix config."""

    def __init__(self, message, status=None, retryable=False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def top_confidence(probabilities):
    """How far the top option stands above a uniform guess: (n * p_max - 1) / (n - 1).

    0 for a uniform distribution, 1 for certainty. Computed here from the probabilities so
    thresholds don't depend on which confidence measure a server reports.
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, (n * max(probabilities.values()) - 1) / (n - 1))


class LayaClient:
    def __init__(self, url=None, model=None, api_key=None, timeout=1.5, max_retries=1):
        base = (url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip('/')
        self.url = base if base.endswith('/v1/systemone') else base + '/v1/systemone'
        self.model = model or os.environ.get(MODEL_ENV)   # None lets Laya route by language
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        self.timeout = timeout
        self.max_retries = max_retries

    def decide(self, state, questions):
        payload = {'state': state, 'questions': questions}
        if self.model:
            payload['model'] = self.model
        body = json.dumps(payload).encode()
        headers = {'Content-Type': 'application/json', 'User-Agent': 'create3-laya/0.1'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(self.url, data=body, headers=headers, method='POST')
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors='replace')[:300]
                retryable = e.code in RETRYABLE_STATUSES
                if retryable and attempt < self.max_retries:
                    time.sleep(0.1 * 2 ** attempt)
                    continue
                raise LayaError(f'HTTP {e.code} from {self.url}: {detail}', status=e.code,
                                retryable=retryable)
            except (urllib.error.URLError, OSError) as e:
                if attempt < self.max_retries:
                    time.sleep(0.1 * 2 ** attempt)
                    continue
                raise LayaError(f'cannot reach {self.url}: {e}', retryable=True)
            except json.JSONDecodeError as e:
                raise LayaError(f'response from {self.url} was not JSON: {e}', retryable=True)
