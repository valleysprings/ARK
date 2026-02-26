#!/usr/bin/env python3
"""
Token Cost Tracker for Answer Augmented Retrieval

Provides precise token counting and cost calculation for LLM invocations using tokencost library.
Tracks token usage and costs per operation and session.
"""

import warnings
import os
import sys
import contextlib
import contextvars
from typing import List, Dict, Any, Union, Optional
from decimal import Decimal
from datetime import datetime
from io import StringIO

# Per-document context variable for concurrent tracking
_current_doc_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('current_doc_id', default=None)


def set_current_doc_id(doc_id: str):
    """Set the current document ID for per-doc token tracking (asyncio-safe)."""
    _current_doc_id.set(doc_id)


def get_current_doc_id() -> Optional[str]:
    """Get the current document ID."""
    return _current_doc_id.get()

# Comprehensive warning suppression for tiktoken model update messages
# Must be done BEFORE importing tokencost/tiktoken
warnings.filterwarnings("ignore", message=".*may update over time.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Returning num tokens assuming.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*gpt-4.*may update over time.*", category=UserWarning)

# Also suppress at environment level for tiktoken
os.environ.setdefault('TIKTOKEN_CACHE_DIR', '/tmp/tiktoken_cache')


# Context manager to suppress stderr output
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


# Import tiktoken with both warnings and stderr suppressed
with warnings.catch_warnings(), suppress_stderr():
    warnings.simplefilter("ignore", UserWarning)
    try:
        import tiktoken
        import tokencost
        TOKENCOST_AVAILABLE = True
    except ImportError:
        TOKENCOST_AVAILABLE = False
        print("Warning: tokencost not available. Install with: pip install tokencost")


# Global token tracking across all operations
_global_session_stats = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_prompt_cost": Decimal('0.0'),
    "total_completion_cost": Decimal('0.0'),
    "operation_stats": {},
    "call_count": 0
}


class TokenCostTracker:
    """
    Track token usage and costs for LLM invocations

    This tracker integrates with the LLM client to monitor token consumption
    and associated costs across different models and operations.
    """

    def __init__(self, model_name: str, enable_logging: bool = True):
        """
        Initialize token cost tracker

        Args:
            model_name: Name of the model used for cost calculation
            enable_logging: Whether to print cost summaries (default: True)
        """
        self.model_name = model_name
        self.enable_logging = enable_logging

        # Initialize session stats first (always needed)
        self.session_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_prompt_cost": Decimal('0.0'),
            "total_completion_cost": Decimal('0.0'),
            "operation_stats": {},
            "call_count": 0
        }

        if not TOKENCOST_AVAILABLE:
            if enable_logging:
                print(f"Warning: tokencost not available. Token tracking disabled.")
            self.cost_model = None
            return

        # Validate model is supported by tokencost
        if model_name not in tokencost.TOKEN_COSTS:
            # Try to find closest match or use default
            available_models = list(tokencost.TOKEN_COSTS.keys())
            if enable_logging:
                print(f"Warning: Model '{model_name}' not found in tokencost.")

            # Use gpt-4o as fallback for cost calculation
            self.cost_model = "gpt-4o"
            if enable_logging:
                print(f"Using '{self.cost_model}' for cost calculation as fallback")
        else:
            self.cost_model = model_name

    def count_tokens(self, text: Union[str, List[Dict]], is_messages: bool = False) -> int:
        """
        Count tokens in text or message list

        Args:
            text: String or list of message dictionaries
            is_messages: True if text is a list of message dictionaries

        Returns:
            Number of tokens
        """
        if not TOKENCOST_AVAILABLE or self.cost_model is None:
            # Fallback to word count estimation (1 token ≈ 0.75 words)
            word_count = len(str(text).split())
            return int(word_count / 0.75)

        try:
            # Suppress all tiktoken related warnings and stderr output for cleaner output
            with warnings.catch_warnings(), suppress_stderr():
                warnings.filterwarnings("ignore", message=".*may update over time.*")
                warnings.filterwarnings("ignore", message=".*Returning num tokens assuming.*")
                warnings.filterwarnings("ignore", category=UserWarning, module="tiktoken")

                # Map unsupported model names to tiktoken-supported equivalents
                # gpt-5, gpt-4o, o-series models all use o200k_base encoding
                tiktoken_model = self.cost_model
                if self.cost_model.startswith('gpt-5') or self.cost_model.startswith('o3-') or self.cost_model.startswith('o4-'):
                    tiktoken_model = 'gpt-4o'  # gpt-4o uses o200k_base in tiktoken

                # For gpt-5 models, use tiktoken directly to avoid warnings
                if self.cost_model.startswith('gpt-5') or self.cost_model.startswith('o3-') or self.cost_model.startswith('o4-'):
                    encoding = tiktoken.get_encoding("o200k_base")
                    if is_messages and isinstance(text, list):
                        # Count tokens for message format
                        num_tokens = 0
                        for message in text:
                            num_tokens += 4  # Message overhead
                            for key, value in message.items():
                                num_tokens += len(encoding.encode(str(value)))
                        num_tokens += 2  # Reply overhead
                        return num_tokens
                    else:
                        return len(encoding.encode(str(text)))
                else:
                    if is_messages and isinstance(text, list):
                        try:
                            return tokencost.count_message_tokens(text, tiktoken_model)
                        except KeyError:
                            return tokencost.count_message_tokens(text, "gpt-4o")
                    else:
                        return tokencost.count_string_tokens(str(text), tiktoken_model)
        except Exception as e:
            if self.enable_logging:
                print(f"Error counting tokens: {e}")
            # Fallback to word count estimation
            word_count = len(str(text).split())
            return int(word_count / 0.75)

    def calculate_prompt_cost(self, prompt: Union[str, List[Dict]]) -> Decimal:
        """
        Calculate cost for prompt

        Args:
            prompt: String prompt or list of message dictionaries

        Returns:
            Cost in USD as Decimal
        """
        if not TOKENCOST_AVAILABLE or self.cost_model is None:
            return Decimal('0.0')

        try:
            with warnings.catch_warnings(), suppress_stderr():
                warnings.filterwarnings("ignore", message=".*may update over time.*")
                warnings.filterwarnings("ignore", message=".*Returning num tokens assuming.*")
                warnings.filterwarnings("ignore", category=UserWarning, module="tiktoken")

                # For gpt-5 models, calculate cost manually using o200k_base encoding
                if self.cost_model.startswith('gpt-5') or self.cost_model.startswith('o3-') or self.cost_model.startswith('o4-'):
                    # Count tokens using o200k_base
                    token_count = self.count_tokens(prompt, is_messages=isinstance(prompt, list))
                    # Get cost per token from tokencost
                    if self.cost_model in tokencost.TOKEN_COSTS:
                        cost_per_token = Decimal(str(tokencost.TOKEN_COSTS[self.cost_model]['input_cost_per_token']))
                        return cost_per_token * token_count
                    return Decimal('0.0')
                else:
                    # Use cost_model directly for cost calculation (tokencost supports the model)
                    try:
                        return tokencost.calculate_prompt_cost(prompt, self.cost_model)
                    except KeyError:
                        return tokencost.calculate_prompt_cost(prompt, "gpt-4o")
        except Exception as e:
            if self.enable_logging:
                print(f"Error calculating prompt cost: {e}")
            return Decimal('0.0')

    def calculate_completion_cost(self, completion: str) -> Decimal:
        """
        Calculate cost for completion/response

        Args:
            completion: Response text from the model

        Returns:
            Cost in USD as Decimal
        """
        if not TOKENCOST_AVAILABLE or self.cost_model is None:
            return Decimal('0.0')

        try:
            with warnings.catch_warnings(), suppress_stderr():
                warnings.filterwarnings("ignore", message=".*may update over time.*")
                warnings.filterwarnings("ignore", message=".*Returning num tokens assuming.*")
                warnings.filterwarnings("ignore", category=UserWarning, module="tiktoken")

                # For gpt-5 models, calculate cost manually using o200k_base encoding
                if self.cost_model.startswith('gpt-5') or self.cost_model.startswith('o3-') or self.cost_model.startswith('o4-'):
                    # Count tokens using o200k_base
                    token_count = self.count_tokens(completion)
                    # Get cost per token from tokencost
                    if self.cost_model in tokencost.TOKEN_COSTS:
                        cost_per_token = Decimal(str(tokencost.TOKEN_COSTS[self.cost_model]['output_cost_per_token']))
                        return cost_per_token * token_count
                    return Decimal('0.0')
                else:
                    # Use cost_model directly for cost calculation (tokencost supports the model)
                    return tokencost.calculate_completion_cost(completion, self.cost_model)
        except Exception as e:
            if self.enable_logging:
                print(f"Error calculating completion cost: {e}")
            return Decimal('0.0')

    def calculate_total_cost(self, prompt: Union[str, List[Dict]], response: str) -> Dict[str, Any]:
        """
        Calculate total cost for a conversation (prompt + response)

        Args:
            prompt: String prompt or list of message dictionaries
            response: Response text from the model

        Returns:
            Dictionary with detailed token and cost information
        """
        try:
            # Count tokens
            prompt_tokens = self.count_tokens(prompt, is_messages=isinstance(prompt, list))
            response_tokens = self.count_tokens(response)
            total_tokens = prompt_tokens + response_tokens

            # Calculate costs
            prompt_cost = self.calculate_prompt_cost(prompt)
            response_cost = self.calculate_completion_cost(response)
            total_cost = prompt_cost + response_cost

            return {
                "model_used": self.model_name,
                "cost_model": self.cost_model,
                "tokens": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": response_tokens,
                    "total_tokens": total_tokens
                },
                "costs_usd": {
                    "prompt_cost": float(prompt_cost),
                    "completion_cost": float(response_cost),
                    "total_cost": float(total_cost)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            if self.enable_logging:
                print(f"Error calculating total cost: {e}")
            return {
                "model_used": self.model_name,
                "cost_model": self.cost_model,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "costs_usd": {"prompt_cost": 0.0, "completion_cost": 0.0, "total_cost": 0.0},
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def track_call(self, operation_name: str, prompt: Union[str, List[Dict]], response: str) -> Dict[str, Any]:
        """
        Track a complete LLM call with cost calculation and session statistics

        Args:
            operation_name: Name of the operation (e.g., "entity_extraction", "summarization")
            prompt: String prompt or list of message dictionaries
            response: Response text from the model

        Returns:
            Dictionary with cost information and updated session stats
        """
        global _global_session_stats

        # Calculate costs for this call
        cost_info = self.calculate_total_cost(prompt, response)

        # Update global session statistics
        _global_session_stats["total_prompt_tokens"] += cost_info["tokens"]["prompt_tokens"]
        _global_session_stats["total_completion_tokens"] += cost_info["tokens"]["completion_tokens"]
        _global_session_stats["total_prompt_cost"] += Decimal(str(cost_info["costs_usd"]["prompt_cost"]))
        _global_session_stats["total_completion_cost"] += Decimal(str(cost_info["costs_usd"]["completion_cost"]))
        _global_session_stats["call_count"] += 1

        # Update operation-specific statistics in global tracker
        if operation_name not in _global_session_stats["operation_stats"]:
            _global_session_stats["operation_stats"][operation_name] = {
                "calls": 0,
                "total_tokens": 0,
                "total_cost": Decimal('0.0')
            }

        op_stats = _global_session_stats["operation_stats"][operation_name]
        op_stats["calls"] += 1
        op_stats["total_tokens"] += cost_info["tokens"]["total_tokens"]
        op_stats["total_cost"] += Decimal(str(cost_info["costs_usd"]["total_cost"]))

        # Also update local session stats for backward compatibility
        self.session_stats["total_prompt_tokens"] += cost_info["tokens"]["prompt_tokens"]
        self.session_stats["total_completion_tokens"] += cost_info["tokens"]["completion_tokens"]
        self.session_stats["total_prompt_cost"] += Decimal(str(cost_info["costs_usd"]["prompt_cost"]))
        self.session_stats["total_completion_cost"] += Decimal(str(cost_info["costs_usd"]["completion_cost"]))
        self.session_stats["call_count"] += 1

        if operation_name not in self.session_stats["operation_stats"]:
            self.session_stats["operation_stats"][operation_name] = {
                "calls": 0,
                "total_tokens": 0,
                "total_cost": Decimal('0.0')
            }

        local_op_stats = self.session_stats["operation_stats"][operation_name]
        local_op_stats["calls"] += 1
        local_op_stats["total_tokens"] += cost_info["tokens"]["total_tokens"]
        local_op_stats["total_cost"] += Decimal(str(cost_info["costs_usd"]["total_cost"]))

        # Log the costs if enabled
        if self.enable_logging:
            print(
                f"[Token Cost] {operation_name}: {cost_info['tokens']['total_tokens']} tokens, "
                f"${cost_info['costs_usd']['total_cost']:.6f}"
            )

        # Add operation name to cost info
        cost_info["operation_name"] = operation_name

        return cost_info

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive session cost summary from global statistics

        Returns:
            Dictionary with session cost statistics
        """
        global _global_session_stats

        total_cost = _global_session_stats["total_prompt_cost"] + _global_session_stats["total_completion_cost"]
        total_tokens = _global_session_stats["total_prompt_tokens"] + _global_session_stats["total_completion_tokens"]

        # Convert operation stats to serializable format
        operation_summary = {}
        for operation, stats in _global_session_stats["operation_stats"].items():
            operation_summary[operation] = {
                "calls": stats["calls"],
                "total_tokens": stats["total_tokens"],
                "total_cost_usd": float(stats["total_cost"]),
                "avg_tokens_per_call": stats["total_tokens"] / stats["calls"] if stats["calls"] > 0 else 0,
                "avg_cost_per_call": float(stats["total_cost"]) / stats["calls"] if stats["calls"] > 0 else 0.0
            }

        return {
            "model_used": self.model_name,
            "cost_model": self.cost_model,
            "session_totals": {
                "calls": _global_session_stats["call_count"],
                "total_tokens": total_tokens,
                "prompt_tokens": _global_session_stats["total_prompt_tokens"],
                "completion_tokens": _global_session_stats["total_completion_tokens"],
                "total_cost_usd": float(total_cost),
                "prompt_cost_usd": float(_global_session_stats["total_prompt_cost"]),
                "completion_cost_usd": float(_global_session_stats["total_completion_cost"]),
                "avg_tokens_per_call": total_tokens / _global_session_stats["call_count"] if _global_session_stats["call_count"] > 0 else 0,
                "avg_cost_per_call": float(total_cost) / _global_session_stats["call_count"] if _global_session_stats["call_count"] > 0 else 0.0
            },
            "operation_breakdown": operation_summary,
            "timestamp": datetime.now().isoformat()
        }

    def log_session_summary(self):
        """Log final session cost summary"""
        if not self.enable_logging:
            return

        summary = self.get_session_summary()

        session_totals = summary["session_totals"]
        print("\n" + "="*80)
        print("SESSION COST SUMMARY")
        print("="*80)
        print(f"Total Cost: ${session_totals['total_cost_usd']:.6f}")
        print(f"Total Tokens: {session_totals['total_tokens']:,}")
        print(f"  - Prompt Tokens: {session_totals['prompt_tokens']:,} (${session_totals['prompt_cost_usd']:.6f})")
        print(f"  - Completion Tokens: {session_totals['completion_tokens']:,} (${session_totals['completion_cost_usd']:.6f})")
        print(f"Total Calls: {session_totals['calls']}")
        print(f"Average per Call: {session_totals['avg_tokens_per_call']:.0f} tokens, ${session_totals['avg_cost_per_call']:.6f}")
        print()

        # Log per-operation breakdown
        if summary["operation_breakdown"]:
            print("Operation Breakdown:")
            print("-" * 80)
            for operation, stats in summary["operation_breakdown"].items():
                print(f"  {operation}:")
                print(f"    Calls: {stats['calls']}, Tokens: {stats['total_tokens']:,}, Cost: ${stats['total_cost_usd']:.6f}")
                print(f"    Avg: {stats['avg_tokens_per_call']:.0f} tokens/call, ${stats['avg_cost_per_call']:.6f}/call")
        print("="*80 + "\n")


def get_available_models() -> List[str]:
    """Get list of models supported by tokencost"""
    if not TOKENCOST_AVAILABLE:
        return []
    return list(tokencost.TOKEN_COSTS.keys())


def is_model_supported(model_name: str) -> bool:
    """Check if a model is supported by tokencost"""
    if not TOKENCOST_AVAILABLE:
        return False
    return model_name in tokencost.TOKEN_COSTS


def reset_global_session_stats():
    """Reset global session statistics for a new session"""
    global _global_session_stats
    _global_session_stats.update({
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_prompt_cost": Decimal('0.0'),
        "total_completion_cost": Decimal('0.0'),
        "operation_stats": {},
        "call_count": 0
    })


def get_global_session_summary() -> Dict[str, Any]:
    """
    Get the global session summary directly

    Returns:
        Dictionary with global session statistics
    """
    global _global_session_stats

    total_cost = _global_session_stats["total_prompt_cost"] + _global_session_stats["total_completion_cost"]
    total_tokens = _global_session_stats["total_prompt_tokens"] + _global_session_stats["total_completion_tokens"]

    return {
        "session_totals": {
            "calls": _global_session_stats["call_count"],
            "total_tokens": total_tokens,
            "prompt_tokens": _global_session_stats["total_prompt_tokens"],
            "completion_tokens": _global_session_stats["total_completion_tokens"],
            "total_cost_usd": float(total_cost),
            "prompt_cost_usd": float(_global_session_stats["total_prompt_cost"]),
            "completion_cost_usd": float(_global_session_stats["total_completion_cost"]),
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Simple Token Tracker (for direct token counting from API responses)
# ============================================================================

class SimpleTokenTracker:
    """
    Simple token tracker that counts input/output tokens from API responses

    This is a lightweight alternative to TokenCostTracker that works directly
    with token counts from API responses (usage.prompt_tokens, usage.completion_tokens).
    """

    def __init__(self):
        """Initialize the simple token tracker"""
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.call_details = []
        self.per_doc_stats: Dict[str, Dict[str, int]] = {}

    def add_call(self, input_tokens: int, output_tokens: int, model: str = "unknown", operation: str = ""):
        """
        Record a single API call with its token usage

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            model: Model name used for the call
            operation: Optional operation name (e.g., "entity_extraction")
        """
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)
        self.call_details.append({
            "call": self.total_calls,
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        })

        # Track per-document stats via contextvars (asyncio-safe)
        doc_id = _current_doc_id.get()
        if doc_id is not None:
            if doc_id not in self.per_doc_stats:
                self.per_doc_stats[doc_id] = {
                    "total_calls": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                }
            s = self.per_doc_stats[doc_id]
            s["total_calls"] += 1
            s["total_input_tokens"] += input_tokens
            s["total_output_tokens"] += output_tokens
            s["total_tokens"] += (input_tokens + output_tokens)

    def print_summary(self):
        """Print a summary of total token usage"""
        print("\n" + "="*80)
        print("TOKEN USAGE SUMMARY")
        print("="*80)
        print(f"Total API Calls: {self.total_calls}")
        print(f"Total Input Tokens: {self.total_input_tokens:,}")
        print(f"Total Output Tokens: {self.total_output_tokens:,}")
        print(f"Total Tokens: {self.total_tokens:,}")
        if self.total_calls > 0:
            print(f"Average Tokens per Call: {self.total_tokens / self.total_calls:.2f}")
        print("="*80 + "\n")

    def print_details(self):
        """Print detailed breakdown of each call"""
        print("\n" + "="*80)
        print("DETAILED CALL BREAKDOWN")
        print("="*80)
        for detail in self.call_details:
            op_str = f" - {detail['operation']}" if detail['operation'] else ""
            print(f"Call #{detail['call']} [{detail['model']}]{op_str}:")
            print(f"  Input Tokens:  {detail['input_tokens']:,}")
            print(f"  Output Tokens: {detail['output_tokens']:,}")
            print(f"  Total Tokens:  {detail['total_tokens']:,}")
            print("-" * 80)
        print("="*80 + "\n")

    def get_doc_stats(self, doc_id: str) -> Dict[str, int]:
        """Get per-document token stats tracked via contextvars."""
        return self.per_doc_stats.get(doc_id, {
            "total_calls": 0, "total_input_tokens": 0,
            "total_output_tokens": 0, "total_tokens": 0,
        })

    def snapshot(self) -> Dict[str, int]:
        """Return a snapshot of current totals for computing deltas"""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }

    @staticmethod
    def delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
        """Compute the difference between two snapshots"""
        return {k: after[k] - before[k] for k in before}

    def reset(self):
        """Reset all counters and details"""
        self.__init__()


# Global simple token tracker instance
_simple_token_tracker = SimpleTokenTracker()


def get_simple_tracker() -> SimpleTokenTracker:
    """Get the global simple token tracker instance"""
    return _simple_token_tracker


def print_token_summary():
    """Print token usage summary from the global tracker"""
    _simple_token_tracker.print_summary()


def print_token_details():
    """Print detailed token usage from the global tracker"""
    _simple_token_tracker.print_details()


def reset_token_tracker():
    """Reset the global simple token tracker"""
    _simple_token_tracker.reset()
