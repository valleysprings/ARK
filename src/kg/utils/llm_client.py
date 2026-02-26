"""
LLM Client Manager for Answer Augmented Retrieval

Manages multiple LLM providers (DeepSeek, GPT, Google Gemini) with unified interface.
Integrates token tracking and provides retry logic for robust API calls.
"""

import asyncio
import functools
import requests
import http.client
import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    print("Warning: openai package not installed")

try:
    from google import genai
except ImportError:
    genai = None
    print("Warning: google.genai package not installed")

try:
    from .token_tracker import TokenCostTracker, get_simple_tracker, print_token_summary
except ImportError:
    try:
        from token_tracker import TokenCostTracker, get_simple_tracker, print_token_summary
    except ImportError:
        # Fallback if token_tracker not available
        TokenCostTracker = None
        get_simple_tracker = None
        print_token_summary = None


# ============================================================================
# Configuration Loading
# ============================================================================

def load_llm_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load LLM API configuration from YAML file and .env

    Loads configuration from llm.yaml and merges with credentials from .env file

    Args:
        config_path: Path to config file. Defaults to src/config/llm.yaml

    Returns:
        Dictionary with LLM API configuration merged with credentials from .env
    """
    # Load .env file from project root
    if load_dotenv is not None:
        # From kg/utils/llm_client.py, go up 3 levels to project root
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            print(f"Warning: .env file not found at {env_path}")
            print("Please copy .env.example to .env and fill in your API keys")

    if config_path is None:
        # Default to llm.yaml in src/config/
        # From kg/utils/llm_client.py, go up 2 levels to src/, then to config/
        config_path = Path(__file__).parent.parent.parent / "config" / "llm.yaml"

    # Load main configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llm_config = config.get('llm_api', {})

    # Merge credentials from environment variables
    if 'deepseek' in llm_config:
        llm_config['deepseek']['base_url'] = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        llm_config['deepseek']['api_key'] = os.getenv('DEEPSEEK_API_KEY', '')

    if 'gpt' in llm_config:
        llm_config['gpt']['base_url'] = os.getenv('GPT_BASE_URL', 'https://new.gptgod.cloud/v1/')
        llm_config['gpt']['api_key'] = os.getenv('GPT_API_KEY', '')

    if 'gemini' in llm_config:
        llm_config['gemini']['api_key'] = os.getenv('GEMINI_API_KEY', '')

    if 'ollama' in llm_config:
        llm_config['ollama']['base_url'] = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

    if 'vllm' in llm_config:
        llm_config['vllm']['base_url'] = os.getenv('VLLM_BASE_URL', llm_config['vllm'].get('base_url', 'http://localhost:8000/v1'))
        llm_config['vllm']['model'] = os.getenv('VLLM_MODEL', llm_config['vllm'].get('model', ''))

    return llm_config


# Load configuration on module import
_LLM_CONFIG = load_llm_config()

# Extract configuration values directly from YAML (no environment variable fallback)
DEEPSEEK_CONFIG = _LLM_CONFIG.get('deepseek', {})
DEEPSEEK_API_KEY = DEEPSEEK_CONFIG.get('api_key', '')
DEEPSEEK_BASE_URL = DEEPSEEK_CONFIG.get('base_url', 'https://api.deepseek.com')
DEEPSEEK_MODEL = DEEPSEEK_CONFIG['model'] if DEEPSEEK_CONFIG else 'deepseek-chat'

GPT_CONFIG = _LLM_CONFIG.get('gpt', {})
GPT_API_KEY = GPT_CONFIG.get('api_key', '')
GPT_BASE_URL = GPT_CONFIG.get('base_url', 'https://new.gptgod.cloud/v1/')
GPT_MODEL = GPT_CONFIG['model'] if GPT_CONFIG else 'gemini-2.5-flash'

GEMINI_CONFIG = _LLM_CONFIG.get('gemini', {})
GOOGLE_API_KEY = GEMINI_CONFIG.get('api_key', '')
GOOGLE_MODEL = GEMINI_CONFIG['model'] if GEMINI_CONFIG else 'gemini-2.5-flash'

OLLAMA_CONFIG = _LLM_CONFIG.get('ollama', {})
OLLAMA_BASE_URL = OLLAMA_CONFIG.get('base_url', 'http://localhost:11434')
OLLAMA_MODEL = OLLAMA_CONFIG['model'] if OLLAMA_CONFIG else 'mistral:latest'

VLLM_CONFIG = _LLM_CONFIG.get('vllm', {})
VLLM_BASE_URL = VLLM_CONFIG.get('base_url', 'http://localhost:8000/v1')
VLLM_MODEL = VLLM_CONFIG.get('model', '')
VLLM_TIMEOUT = float(VLLM_CONFIG['timeout'] if VLLM_CONFIG else 600)
VLLM_MAX_RETRIES = int(VLLM_CONFIG['max_retries'] if VLLM_CONFIG else 3)

# General settings
SETTINGS = _LLM_CONFIG.get('settings', {})
TIMEOUT = float(DEEPSEEK_CONFIG['timeout'] if DEEPSEEK_CONFIG else SETTINGS['max_async_calls'])
MAX_ASYNC_CALL_SIZE = SETTINGS['max_async_calls']
MAX_RETRIES = DEEPSEEK_CONFIG['max_retries'] if DEEPSEEK_CONFIG else 2
RETRY_DELAY = SETTINGS['retry_delay']


# ============================================================================
# Utility Decorators
# ============================================================================

def limit_async_func_call(max_size: int = None, waiting_time: float = 0.0001):
    """
    Decorator to limit the number of concurrent async function calls.
    If max_size is None, reads MAX_ASYNC_CALL_SIZE at runtime (supports dynamic override).
    """
    def decorator(func):
        __current_size = 0

        @functools.wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            _max = max_size if max_size is not None else MAX_ASYNC_CALL_SIZE
            while __current_size >= _max:
                await asyncio.sleep(waiting_time)
            __current_size += 1
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                __current_size -= 1

        return wait_func

    return decorator


def retry_with_timeout(max_retries=None, timeout=None, delay=None):
    """
    Decorator to add retry logic with exponential backoff to async functions

    Handles common network errors and timeouts with automatic retry.

    Args:
        max_retries: Maximum number of retry attempts (defaults to config value)
        timeout: Timeout in seconds for each attempt (defaults to config value)
        delay: Initial delay between retries (defaults to config value)

    Returns:
        Decorated async function with retry logic
    """
    # Use config values if not specified
    _max_retries = max_retries if max_retries is not None else MAX_RETRIES
    _timeout = timeout if timeout is not None else TIMEOUT
    _delay = delay if delay is not None else RETRY_DELAY

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(_max_retries):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=_timeout
                    )
                except (asyncio.TimeoutError,
                        requests.exceptions.ProxyError,
                        http.client.RemoteDisconnected,
                        ConnectionError) as e:
                    if attempt == _max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1}/{_max_retries} failed: {str(e)}")
                    await asyncio.sleep(_delay * (2 ** attempt))  # Exponential backoff
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# LLM Client Manager
# ============================================================================

class LLMClientManager:
    """
    Unified manager for multiple LLM providers

    Provides a consistent interface across DeepSeek, GPT, and Google Gemini APIs.
    Includes built-in token tracking and retry logic.
    """

    def __init__(self, enable_token_tracking: bool = True):
        """
        Initialize LLM Client Manager

        Args:
            enable_token_tracking: Whether to enable token cost tracking
        """
        self.enable_token_tracking = enable_token_tracking

        # Initialize token trackers for each model
        self.token_trackers = {}
        if enable_token_tracking:
            self.token_trackers = {
                "deepseek": TokenCostTracker(DEEPSEEK_MODEL, enable_logging=True),
                "gpt": TokenCostTracker(GPT_MODEL, enable_logging=True),
                "google": TokenCostTracker(GOOGLE_MODEL, enable_logging=True),
            }

        # Initialize Google client (synchronous initialization)
        if genai is not None:
            self.google_client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            self.google_client = None

    def _track_call(self, provider: str, operation: str, prompt: str, response: str):
        """
        Track token usage for an LLM call

        Args:
            provider: LLM provider name ("deepseek", "gpt", "google")
            operation: Operation name for tracking
            prompt: Input prompt
            response: Model response
        """
        if self.enable_token_tracking and provider in self.token_trackers:
            self.token_trackers[provider].track_call(operation, prompt, response)

    async def call_deepseek(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List = None,
        operation_name: str = "deepseek_call",
        **kwargs
    ) -> str:
        """
        Call DeepSeek API

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            operation_name: Name for token tracking
            **kwargs: Additional arguments for the API call

        Returns:
            Model response as string
        """
        if history_messages is None:
            history_messages = []

        # Create DeepSeek client (per-call to avoid connection issues)
        deepseek_client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Make API call
        response = await deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            **kwargs
        )

        result = response.choices[0].message.content

        # Track tokens from API response
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = input_tokens + output_tokens

            # Add to simple tracker
            tracker = get_simple_tracker()
            tracker.add_call(input_tokens, output_tokens, DEEPSEEK_MODEL, operation_name)

            # Print immediate feedback
            print(f"[Token Usage] Input: {input_tokens:,}, Output: {output_tokens:,}, Total: {total_tokens:,}", flush=True)

        # Track with cost tracker (if tokencost available)
        self._track_call("deepseek", operation_name, prompt, result)

        return result

    @retry_with_timeout()
    async def call_gpt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List = None,
        operation_name: str = "gpt_call",
        **kwargs
    ) -> str:
        """
        Call GPT API via proxy

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            operation_name: Name for token tracking
            **kwargs: Additional arguments for the API call

        Returns:
            Model response as string
        """
        if history_messages is None:
            history_messages = []

        # Create OpenAI client (per-call to avoid connection issues)
        openai_client = AsyncOpenAI(
            api_key=GPT_API_KEY,
            base_url=GPT_BASE_URL,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES
        )

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Make API call using configured GPT_MODEL
        response = await openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            **kwargs
        )

        result = response.choices[0].message.content

        # Track tokens from API response
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = input_tokens + output_tokens

            # Add to simple tracker
            tracker = get_simple_tracker()
            tracker.add_call(input_tokens, output_tokens, GPT_MODEL, operation_name)

            # Print immediate feedback
            print(f"[Token Usage] Input: {input_tokens:,}, Output: {output_tokens:,}, Total: {total_tokens:,}", flush=True)

        # Track with cost tracker (if tokencost available)
        self._track_call("gpt", operation_name, prompt, result)

        return result

    @limit_async_func_call(waiting_time=0.01)
    @retry_with_timeout()
    async def call_google(
        self,
        prompt: str,
        operation_name: str = "google_call",
        **kwargs
    ) -> str:
        """
        Call Google Gemini API

        Args:
            prompt: User prompt
            operation_name: Name for token tracking
            **kwargs: Additional arguments for the API call

        Returns:
            Model response as string
        """
        # Make API call using configured model
        response = await self.google_client.aio.models.generate_content(
            model=GOOGLE_MODEL,
            contents=prompt
        )

        result = response.text

        # Track tokens from API response (Google uses usage_metadata)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            total_tokens = input_tokens + output_tokens

            # Add to simple tracker
            tracker = get_simple_tracker()
            tracker.add_call(input_tokens, output_tokens, GOOGLE_MODEL, operation_name)

            # Print immediate feedback
            print(f"[Token Usage] Input: {input_tokens:,}, Output: {output_tokens:,}, Total: {total_tokens:,}", flush=True)

        # Track with cost tracker (if tokencost available)
        self._track_call("google", operation_name, prompt, result)

        return result

    @limit_async_func_call(waiting_time=0.01)
    @retry_with_timeout(timeout=VLLM_TIMEOUT, max_retries=VLLM_MAX_RETRIES)
    async def call_vllm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List = None,
        operation_name: str = "vllm_call",
        **kwargs
    ) -> str:
        """
        Call local vLLM server (OpenAI-compatible API)
        """
        if history_messages is None:
            history_messages = []

        vllm_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=VLLM_BASE_URL,
            timeout=VLLM_TIMEOUT,
            max_retries=VLLM_MAX_RETRIES
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # Disable thinking for Thinking models to avoid wasting tokens
        extra_body = kwargs.pop("extra_body", {})
        if "thinking" not in VLLM_MODEL.lower():
            pass  # non-thinking model, no need
        else:
            extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

        response = await vllm_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            max_tokens=kwargs.pop("max_tokens", 4096),
            extra_body=extra_body if extra_body else None,
            **kwargs
        )

        result = response.choices[0].message.content

        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = input_tokens + output_tokens

            tracker = get_simple_tracker()
            tracker.add_call(input_tokens, output_tokens, VLLM_MODEL, operation_name)

            print(f"[Token Usage] Input: {input_tokens:,}, Output: {output_tokens:,}, Total: {total_tokens:,}", flush=True)

        return result

    def get_session_summary(self, provider: Optional[str] = None) -> dict:
        """
        Get token usage session summary

        Args:
            provider: Specific provider to get summary for, or None for all

        Returns:
            Dictionary with session statistics
        """
        if not self.enable_token_tracking:
            return {"error": "Token tracking not enabled"}

        if provider and provider in self.token_trackers:
            return self.token_trackers[provider].get_session_summary()

        # Return summary for all providers
        summaries = {}
        for prov, tracker in self.token_trackers.items():
            summaries[prov] = tracker.get_session_summary()
        return summaries

    def log_session_summary(self):
        """Print session summary for all providers"""
        if not self.enable_token_tracking:
            return

        print("\n" + "="*80)
        print("LLM CLIENT SESSION SUMMARY")
        print("="*80)

        for provider, tracker in self.token_trackers.items():
            print(f"\n{provider.upper()} Provider:")
            print("-" * 80)
            tracker.log_session_summary()


# ============================================================================
# Convenience Functions (Backward Compatibility)
# ============================================================================

# Global client instance
_global_llm_client = LLMClientManager(enable_token_tracking=True)


async def deepseek_model(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List = None,
    **kwargs
) -> str:
    """
    Convenience function for DeepSeek API calls

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Optional conversation history
        **kwargs: Additional arguments

    Returns:
        Model response
    """
    return await _global_llm_client.call_deepseek(
        prompt, system_prompt, history_messages, **kwargs
    )


async def gpt_model(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List = None,
    **kwargs
) -> str:
    """
    Convenience function for GPT API calls

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Optional conversation history
        **kwargs: Additional arguments

    Returns:
        Model response
    """
    return await _global_llm_client.call_gpt(
        prompt, system_prompt, history_messages, **kwargs
    )


async def google_model(prompt: str, **kwargs) -> str:
    """
    Convenience function for Google Gemini API calls

    Args:
        prompt: User prompt
        **kwargs: Additional arguments

    Returns:
        Model response
    """
    return await _global_llm_client.call_google(prompt, **kwargs)


async def vllm_model(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List = None,
    **kwargs
) -> str:
    """
    Convenience function for local vLLM server calls

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Optional conversation history
        **kwargs: Additional arguments

    Returns:
        Model response
    """
    return await _global_llm_client.call_vllm(
        prompt, system_prompt, history_messages, **kwargs
    )


def get_llm_session_summary() -> dict:
    """Get session summary from global LLM client"""
    return _global_llm_client.get_session_summary()


def log_llm_session_summary():
    """Log session summary from global LLM client"""
    _global_llm_client.log_session_summary()
