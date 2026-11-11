"""Unified LLM Client with Persistent Caching and Resilient Retries.

Supports Google Gemini (default: gemini-3.1-flash-lite) and OpenAI.
Caches every prompt/response pair by sha256 hash to eliminate redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config

logger = logging.getLogger(__name__)

# Ensure cache directory exists
LLM_CACHE_DIR = config.CACHE_DIR / "llm"
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _compute_cache_key(prompt: str, system_prompt: str, model: str, version: str) -> str:
    payload = f"{model}:{version}:{system_prompt}:{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LLMClient:
    def __init__(self, model_name: Optional[str] = None, cache_enabled: bool = True):
        load_dotenv(config.PROJECT_ROOT / ".env")
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.cache_enabled = cache_enabled
        
        # Collect all Gemini keys (single or comma-separated)
        raw_keys = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
        self.gemini_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.current_key_idx = 0

        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.provider = config.LLM_PROVIDER

        if self.provider == "gemini" and not self.gemini_keys:
            raise ValueError("GEMINI_API_KEY or GEMINI_API_KEYS is not set in environment or .env file.")

    @property
    def gemini_key(self) -> str:
        if not self.gemini_keys:
            return ""
        return self.gemini_keys[self.current_key_idx % len(self.gemini_keys)]

    def rotate_gemini_key(self) -> str:
        if len(self.gemini_keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.gemini_keys)
            logger.info(f"Rotated to Gemini API key index {self.current_key_idx} (total {len(self.gemini_keys)} keys).")
        return self.gemini_key

    def _call_gemini_rest(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.0,
        max_retries: int = 8,
    ) -> str:
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instruction: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        delay = 2.0
        for attempt in range(max_retries):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.gemini_key}"
            try:
                response = requests.post(url, json=body, timeout=45)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                    return ""
                
                # If credits depleted (402) or quota exhausted (429) and multiple keys are available:
                if response.status_code in (402, 429) and len(self.gemini_keys) > 1:
                    logger.warning(
                        f"Key hit status {response.status_code}. Automatically rotating to next available Gemini key..."
                    )
                    self.rotate_gemini_key()
                    time.sleep(1.0)
                    continue

                # Check for rate limit (429) or temporary server busy (500, 503)
                if response.status_code == 429:
                    wait_time = max(delay, 22.0)
                    logger.warning(
                        f"Gemini API rate limit (429). Waiting {wait_time:.1f}s for quota window (attempt {attempt+1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                    delay = min(delay * 2, 45.0)
                    continue
                elif response.status_code in (500, 503):
                    logger.warning(
                        f"Gemini API busy ({response.status_code}). Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                else:
                    raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text}")
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning(f"Connection issue: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        raise RuntimeError(f"Gemini API failed after {max_retries} retries for model {self.model_name}")

    def query(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Queries LLM with local file-level caching and automated retries."""
        cache_key = _compute_cache_key(prompt, system_prompt, self.model_name, config.LLM_CACHE_VERSION)
        cache_file = LLM_CACHE_DIR / f"{cache_key}.json"

        if self.cache_enabled and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    return cached.get("response", "")
            except Exception:
                pass

        if self.provider == "gemini":
            result = self._call_gemini_rest(prompt, system_prompt, json_mode, temperature)
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported.")

        if self.cache_enabled:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "key": cache_key,
                            "model": self.model_name,
                            "prompt": prompt,
                            "system_prompt": system_prompt,
                            "response": result,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                logger.warning(f"Failed to write cache for {cache_key}: {e}")

        return result


_DEFAULT_CLIENT: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LLMClient()
    return _DEFAULT_CLIENT

