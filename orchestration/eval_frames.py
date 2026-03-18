# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------
# NOTE: This file has been heavily modified from the original script (https://github.com/NVlabs/ToolOrchestra/blob/main/evaluation/eval_frames.py)
#        to support skill-based agent orchestration. 
#
# Changes:
#   - Skill-based agent orchestration with different routing strategies
#   - Generate exploration bundles with different agents per mode
#   - Context management: progressive truncation, context filtering, context limits, retry
#   - Cost tracking: token/cost per model (routed + orchestrator)
#   - Multi-model support: GPT-5, Claude, Gemini (API) in addition to vLLM
#   - Resume, concurrency, and other fallback logics
# ------------------------------------------------------------

import os
import random
import time
import json
import requests
import asyncio
import subprocess
from pathlib import Path
from tqdm import tqdm
from dataclasses import asdict
import sys

try:
    from dotenv import load_dotenv
    _so_root = Path(__file__).resolve().parent.parent
    load_dotenv(_so_root / ".env")
except ImportError:
    pass

_ORCHESTRATION_DIR = Path(__file__).resolve().parent
_SO_ROOT = _ORCHESTRATION_DIR.parent
sys.path.insert(0, str(_ORCHESTRATION_DIR))
sys.path.insert(0, str(_SO_ROOT))
from config import MODEL_MAPPING as _SO_MODEL_MAPPING

from LLM_CALL import get_llm_response, ContextLengthExceeded
import multiprocessing as mp
import argparse
import logging
from openai import OpenAI
from typing import Callable, Any, Optional
logging.disable(logging.CRITICAL)

MODEL_NAME = None
my_output_dir = None
MAX_ROUNDS = None
MODEL_TYPE = None
MODEL_MAPPING = None  # Set from config.models, may be overridden by --model_name
TOOL_PRICING = None
vllm_model_configs = None
tool_concurrency = 5  # Default concurrency for tool calls
RETRIEVER_CACHE_DIR = None  # Cache directory for retrieval service

# Skill-based agent orchestration support
ROUTING_STRATEGY = "none"
HANDBOOK = None

IS_HLE = False

# Forced model selection for exploration data generation
FORCE_SEARCH_MODEL = None
FORCE_REASONING_MODEL = None
FORCE_ANSWER_MODEL = None
INJECT_REASONING = False
INJECT_SEARCH = False

# Checkpoint support
SAVE_CHECKPOINT_DIR = None
LOAD_CHECKPOINT_DIR = None
CHECKPOINT_STAGE = None

# Context length limits per model (tokens) - for truncation
MODEL_CONTEXT_LIMITS = {
    "Qwen/Qwen2.5-Math-72B-Instruct": 4000,
    "Qwen/Qwen2.5-Math-7B-Instruct": 4000,
    "Qwen/Qwen3-8B": 40960,
    "Nemotron-Orchestrator-8B": 40960,
    "Qwen/Qwen3-32B": 30000,
    "Qwen/Qwen2.5-Coder-32B-Instruct": 120000,
    "meta-llama/Llama-3.3-70B-Instruct": 120000,
    "gpt-5": 270000,
    "gpt-5-mini": 270000,
    "claude-opus-4-5@20251101": 200000,
    "claude-sonnet-4-5@20250929": 200000,
    "gemini-3-pro": 2000000,
    "gemini-2.5-pro": 2000000,
}
with open(_ORCHESTRATION_DIR / 'tools.json') as f:
    raw_tools = json.load(f)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

MODEL_MAPPING = _SO_MODEL_MAPPING.copy()

TOOL_PRICING = {
    "gpt-5": {
        "input_tokens_per_million": 1.25,
        "output_tokens_per_million": 10
    },
    "gpt-5-mini": {
        "input_tokens_per_million": 0.25,
        "output_tokens_per_million": 2
    },
    "Qwen/Qwen3-32B": {
        "input_tokens_per_million": 0.1,
        "output_tokens_per_million": 0.8
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "input_tokens_per_million": 0.1,
        "output_tokens_per_million": 0.8
    },
    "Qwen/Qwen2.5-Math-72B-Instruct": {
        "input_tokens_per_million": 0.1125,
        "output_tokens_per_million": 0.9
    },
    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "input_tokens_per_million": 0.025,
        "output_tokens_per_million": 0.2
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "input_tokens_per_million": 0.1125,
        "output_tokens_per_million": 0.9
    },
    "Qwen/Qwen3-8B": {
        "input_tokens_per_million": 0.025,
        "output_tokens_per_million": 0.2
    },
    "Nemotron-Orchestrator-8B": {
        "input_tokens_per_million": 0.025,
        "output_tokens_per_million": 0.2
    },
    "claude-opus-4-5@20251101": {
        "input_tokens_per_million": 5.0,
        "output_tokens_per_million": 25.0
    },
    "claude-sonnet-4-5@20250929": {
        "input_tokens_per_million": 3.0,
        "output_tokens_per_million": 15.0
    },
    "gemini-3-pro": {
        "input_tokens_per_million": 2.0,
        "output_tokens_per_million": 12.0
    },
    "gemini-3-pro-preview": {
        "input_tokens_per_million": 2.0,
        "output_tokens_per_million": 12.0
    },
    "gemini-2.5-pro": {
        "input_tokens_per_million": 1.25,
        "output_tokens_per_million": 10.0
    },
    "code_interpreter_per_second": 0.0000083,
    "tavily": {
        "search": 0.01,
        "extract": 0.002
    },
}

ALL_TOOLS = {
    "enhance_reasoning": {
        'model': ["reasoner-1", "reasoner-2", "reasoner-3"]
    },
    "code": {  # Alias for enhance_reasoning - some orchestrators may call it "code"
        'model': ["reasoner-1", "reasoner-2", "reasoner-3"]
    },
    "answer": {
        'model': ["answer-math-1", "answer-math-2", "answer-1", "answer-2", "answer-3", "answer-4"]
    },
    "search": {
        "model": ["search-1", "search-2", "search-3"]
    },
}

def cut_seq(seq,l):
    if len(seq)==0:
        return {
            'effective_length': 0,
            'string_after_cut': ''
        }
    token_ids = tokenizer(seq)['input_ids']
    rs = tokenizer.batch_decode(token_ids[-l:], skip_special_tokens=True)
    return {
        'effective_length': len(token_ids),
        'string_after_cut': ''.join(rs)
    }

def estimate_context_tokens(context_str: str, problem: str) -> int:
    """Estimate total token count for context + problem."""
    full_text = context_str + "\n\n" + problem
    # Use tokenizer to get accurate count
    token_ids = tokenizer(full_text)['input_ids']
    return len(token_ids)


def _parse_context_length_error(err_msg: str) -> tuple[int | None, int | None]:
    """Parse vLLM context-length error to extract input_length and max_allowed.
    e.g. 'Input length (30974 tokens) exceeds the maximum allowed length (26242 tokens)'
    Returns (input_length, max_allowed) or (None, None) if unparseable."""
    import re as _re
    m = _re.search(r"Input length \((\d+) tokens\) exceeds the maximum allowed length \((\d+) tokens\)", err_msg, _re.I)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _re.search(r"maximum allowed length \((\d+)", err_msg, _re.I)
    if m:
        return None, int(m.group(1))
    return None, None


def _truncate_context_str_docs(context_str: str, max_tokens: int) -> str:
    """Truncate context_str by shrinking the document section, preserving code.
    Handbook is not in context_str - it's in the prompt template. Returns truncated context."""
    doc_header = "Documents:\n"
    code_header = "\npython code and execution outputs:\n"
    if not context_str.startswith(doc_header):
        return cut_seq(seq=context_str, l=max_tokens)['string_after_cut']
    code_idx = context_str.find(code_header, len(doc_header))
    if code_idx == -1:
        doc_section = context_str[len(doc_header):]
        truncated = cut_seq(seq=doc_section, l=max_tokens)['string_after_cut']
        return doc_header + truncated
    doc_section = context_str[len(doc_header):code_idx]
    code_section = context_str[code_idx + len(code_header):]
    code_tokens = len(tokenizer(code_section)['input_ids'])
    doc_available = max(500, max_tokens - code_tokens)
    doc_truncated = cut_seq(seq=doc_section, l=doc_available)['string_after_cut']
    return doc_header + doc_truncated + code_header + code_section


def filter_models_by_context_length(tools: list, context_str: str, problem: str, skip_answer_tool: bool = False) -> list:
    """
    Filter out models that can't handle the context length.
    
    Args:
        tools: List of tool definitions from tools.json
        context_str: Current context string
        problem: Problem/question string
        skip_answer_tool: If True, skip filtering for answer tool (context will be truncated per model anyway)
        
    Returns:
        Filtered tools list with models that can't handle context removed
    """
    estimated_tokens = estimate_context_tokens(context_str, problem)
    
    filtered_tools = []
    for tool in tools:
        tool_name = tool.get('function', {}).get('name', '')
        if tool_name not in ALL_TOOLS:
            # Keep tool as-is if we don't know about it
            filtered_tools.append(tool)
            continue
        
        # Skip filtering for answer tool if requested (will be filtered later with actual context)
        if skip_answer_tool and tool_name == 'answer':
            filtered_tools.append(tool)  # Keep answer tool as-is
            print(f"[Context Filter] Skipping pre-filtering for {tool_name} (will filter later with actual context)")
            continue
        
        # Get model parameter definition
        model_param = tool.get('function', {}).get('parameters', {}).get('properties', {}).get('model', {})
        if not model_param:
            filtered_tools.append(tool)
            continue
        
        # Get available model choices
        model_enum = model_param.get('enum', [])
        if not model_enum:
            filtered_tools.append(tool)
            continue
        
        # Filter models that can handle the context
        valid_models = []
        for model_alias in model_enum:
            # Map alias to actual model name
            actual_model = MODEL_MAPPING.get(model_alias, model_alias)
            
            # Get context limit for this model
            max_context = MODEL_CONTEXT_LIMITS.get(actual_model, 100000)  # Default: very high
            
            # Check if model can handle the context
            if estimated_tokens <= max_context:
                valid_models.append(model_alias)
            else:
                print(f"[Context Filter] Removing {model_alias} ({actual_model}) from {tool_name}: "
                      f"~{estimated_tokens} tokens > {max_context} limit")
        
        # If no valid models, keep original (let it fail later) or skip tool?
        # For now, keep tool even if no valid models (let orchestrator see it's unavailable)
        if valid_models:
            # Create a copy of the tool with filtered models
            filtered_tool = json.loads(json.dumps(tool))  # Deep copy
            filtered_tool['function']['parameters']['properties']['model']['enum'] = valid_models
            filtered_tools.append(filtered_tool)
        else:
            # No valid models - still include tool but with empty enum (will fail gracefully)
            print(f"[Context Filter] WARNING: No valid models for {tool_name} (all exceed context limit)")
            filtered_tools.append(tool)
    
    return filtered_tools

def retry_api_call(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = None,
    **kwargs
) -> Any:
    """
    Retry wrapper for API calls with exponential backoff.
    
    Args:
        func: The API call function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for exponential backoff
        retryable_exceptions: Tuple of exception types to retry on
        **kwargs: Arguments to pass to func
    
    Returns:
        Result from func
    
    Raises:
        Last exception if all retries fail
    """
    if retryable_exceptions is None:
        from openai import APIError, APITimeoutError, APIConnectionError, RateLimitError
        retryable_exceptions = (
            APIError,
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            TimeoutError,
            ConnectionError,
            requests.exceptions.RequestException,
        )
    
    last_exception = None
    delay = base_delay
    
    for attempt in range(max_retries + 1):
        try:
            return func(**kwargs)
        except retryable_exceptions as e:
            last_exception = e
            
            # Check if it's a rate limit error (429) - use longer delay
            if hasattr(e, 'status_code') and e.status_code == 429:
                # Try to get retry-after header if available
                retry_after = None
                if hasattr(e, 'response') and e.response is not None:
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            pass
                
                # If no retry-after header, use exponential backoff
                if retry_after is None:
                    delay = min(delay * backoff_factor, max_delay)
            else:
                # For other errors, use exponential backoff
                delay = min(delay * backoff_factor, max_delay)
            
            if attempt < max_retries:
                # Log retry attempt
                error_msg = str(e)
                if hasattr(e, 'status_code'):
                    error_msg = f"Status {e.status_code}: {error_msg}"
                print(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                # Last attempt failed
                print(f"API call failed after {max_retries + 1} attempts. Last error: {last_exception}")
                raise last_exception
        except Exception as e:
            # For non-retryable exceptions, raise immediately
            if not isinstance(e, retryable_exceptions):
                raise
            last_exception = e
            if attempt < max_retries:
                delay = min(delay * backoff_factor, max_delay)
                print(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception

def get_llm_response_with_retry(*args, **kwargs):
    """
    Wrapper for get_llm_response with retry logic for API-based models.
    Only applies retry for API-based models (gpt-5, etc.), not vLLM models.
    """
    # Only apply retry for API-based models (not vLLM)
    model_type = kwargs.get('model_type', None)
    model = kwargs.get('model', args[0] if args else None)
    
    # If it's a vLLM model, call directly (vLLM has its own retry logic)
    if model_type == 'vllm' or (model and 'qwen' in model.lower() and 'gpt' not in model.lower()):
        return get_llm_response(*args, **kwargs)
    
    # For API-based models, use retry wrapper
    return retry_api_call(get_llm_response, max_retries=5, *args, **kwargs)

def extract_response_content_and_tool_calls(response, model_name):
    """
    Extract content and tool calls from different response formats.
    
    Handles:
    - OpenAI format (GPT-5): response.choices[0].message.content, response.choices[0].message.tool_calls
    - Claude format: response.content[0].text, response.content blocks with tool_use
    - Gemini format: response.text, response.candidates[0].content.parts with function_calls
    
    Returns:
        (content: str, tool_calls: list or None)
    """
    # OpenAI format (GPT-5, vLLM)
    if hasattr(response, 'choices') and len(response.choices) > 0:
        print(f"[DEBUG] Detected OpenAI format (has 'choices' attribute)")
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = getattr(message, 'tool_calls', None)
        print(f"[DEBUG] OpenAI format: tool_calls attribute exists: {tool_calls is not None}, value: {tool_calls}")
        if tool_calls:
            print(f"[DEBUG] OpenAI format: Found {len(tool_calls)} tool call(s)")
            for i, tc in enumerate(tool_calls):
                tc_name = getattr(tc.function, 'name', 'unknown') if hasattr(tc, 'function') else 'unknown'
                tc_args = getattr(tc.function, 'arguments', '{}') if hasattr(tc, 'function') else '{}'
                print(f"[DEBUG]   Tool call {i+1}: name={tc_name}, arguments={tc_args[:100]}...")
        elif tool_calls is None:
            print(f"[DEBUG] OpenAI format: tool_calls is None (no tool calls in response)")
        else:
            print(f"[DEBUG] OpenAI format: tool_calls is empty list: {tool_calls}")
        return content, tool_calls
    
    # Claude format (Anthropic Message object)
    if hasattr(response, 'content') and not hasattr(response, 'choices'):
        print(f"[DEBUG] Detected Claude format (has 'content' attribute, no 'choices')")
        content_parts = []
        tool_calls = []
        
        print(f"[DEBUG] Claude format: response.content has {len(response.content)} block(s)")
        for idx, block in enumerate(response.content):
            print(f"[DEBUG]   Block {idx}: type={getattr(block, 'type', 'unknown')}, has text={hasattr(block, 'text')}, has name={hasattr(block, 'name')}")
            if hasattr(block, 'text') and block.text:
                content_parts.append(block.text)
            elif hasattr(block, 'type') and block.type == 'tool_use':
                # Claude tool use format - convert to OpenAI-like format
                block_name = getattr(block, 'name', '')
                block_input = getattr(block, 'input', {})
                print(f"[DEBUG] Claude format: Found tool_use block: name={block_name}, input={block_input}")
                tool_call = type('ToolCall', (), {
                    'function': type('Function', (), {
                        'name': block_name,
                        'arguments': json.dumps(block_input)
                    })()
                })()
                tool_calls.append(tool_call)
        
        content = "\n".join(content_parts) if content_parts else ""
        if tool_calls:
            print(f"[DEBUG] Claude format: Converted {len(tool_calls)} tool call(s) to OpenAI format")
        else:
            print(f"[DEBUG] Claude format: No tool_use blocks found in {len(response.content)} content block(s)")
        return content, tool_calls if tool_calls else None
    
    # Gemini format - check for text AND candidates (not choices)
    if hasattr(response, 'candidates') and not hasattr(response, 'choices'):
        print(f"[DEBUG] Detected Gemini format (has 'candidates', no 'choices')")
        # Safely get text - it might be None or raise exception when only function calls are present
        try:
            content = response.text or ""
        except (AttributeError, ValueError, TypeError) as e:
            print(f"[DEBUG] Warning: Could not access response.text (likely function calls only): {e}")
            content = ""
        tool_calls = None
        
        # Check for function_calls attribute directly first (newer Gemini API)
        if hasattr(response, 'function_calls') and response.function_calls:
            print(f"[DEBUG] Gemini format: Found function_calls attribute with {len(response.function_calls)} call(s)")
            function_calls = []
            for func_call in response.function_calls:
                try:
                    func_name = getattr(func_call, 'name', None)
                    func_args = getattr(func_call, 'args', None) or {}
                    if func_name:
                        print(f"[DEBUG] Gemini format: Found function_call: name={func_name}, args={func_args}")
                        tool_call = type('ToolCall', (), {
                            'function': type('Function', (), {
                                'name': func_name,
                                'arguments': json.dumps(func_args) if isinstance(func_args, dict) else (str(func_args) if func_args else "{}")
                            })()
                        })()
                        function_calls.append(tool_call)
                except Exception as e:
                    print(f"WARNING: Failed to parse Gemini function_call: {e}")
                    continue
            if function_calls:
                tool_calls = function_calls
        
        # Fallback: Check for function calls in candidates[0].content.parts
        elif len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                print(f"[DEBUG] Gemini format: Checking candidates[0].content.parts for function calls")
                function_calls = []
                for part in candidate.content.parts:
                    # Check for function_call attribute - be very defensive
                    try:
                        if hasattr(part, 'function_call'):
                            func_call_obj = getattr(part, 'function_call', None)
                            if func_call_obj is not None:
                                # Try to get name and args with multiple fallbacks
                                func_name = None
                                func_args = {}
                                
                                # Try different attribute names for function name
                                for attr_name in ['name', 'function_name', 'functionName']:
                                    if hasattr(func_call_obj, attr_name):
                                        func_name = getattr(func_call_obj, attr_name)
                                        if func_name:
                                            break
                                
                                # Try different attribute names for function args
                                for attr_name in ['args', 'arguments', 'input']:
                                    if hasattr(func_call_obj, attr_name):
                                        func_args = getattr(func_call_obj, attr_name)
                                        if func_args:
                                            break
                                
                                if func_name:
                                    print(f"[DEBUG] Gemini format: Found function_call in parts: name={func_name}, args={func_args}")
                                    # Convert Gemini function call to OpenAI-like format
                                    tool_call = type('ToolCall', (), {
                                        'function': type('Function', (), {
                                            'name': func_name,
                                            'arguments': json.dumps(func_args) if isinstance(func_args, dict) else (str(func_args) if func_args else "{}")
                                        })()
                                    })()
                                    function_calls.append(tool_call)
                    except Exception as e:
                        # Skip this function call if we can't parse it
                        print(f"WARNING: Failed to parse Gemini function call: {e}")
                        continue
                if function_calls:
                    print(f"[DEBUG] Gemini format: Converted {len(function_calls)} function call(s) to OpenAI format")
                    tool_calls = function_calls
        
        if tool_calls:
            print(f"[DEBUG] Gemini format: Final result - {len(tool_calls)} tool call(s) extracted")
        else:
            print(f"[DEBUG] Gemini format: No tool calls found")
        return content, tool_calls
    
    # Fallback: try to get string representation
    if isinstance(response, str):
        print(f"[DEBUG] Response is a string, no tool calls")
        return response, None
    
    # Last resort: return empty
    print(f"[DEBUG] Unknown response format. Response type: {type(response)}, attributes: {dir(response)[:10]}")
    return "", None

def calculate_cost(response, model_name):
    """Calculate cost from response object based on model pricing."""
    if isinstance(response, str):
        return 0.0, 0, 0
    
    # Handle different response formats
    prompt_tokens = 0
    completion_tokens = 0
    
    # Check for usage attribute first
    if hasattr(response, 'usage') and response.usage is not None:
        # Claude format uses input_tokens/output_tokens
        if hasattr(response.usage, 'input_tokens'):
            prompt_tokens = response.usage.input_tokens or 0
            completion_tokens = response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else 0
        # OpenAI format (GPT-5, vLLM) uses prompt_tokens/completion_tokens
        elif hasattr(response.usage, 'prompt_tokens'):
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
    # Gemini format - check for usage_metadata
    elif hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
        prompt_tokens = (response.usage_metadata.prompt_token_count or 0) if hasattr(response.usage_metadata, 'prompt_token_count') else 0
        completion_tokens = (response.usage_metadata.candidates_token_count or 0) if hasattr(response.usage_metadata, 'candidates_token_count') else 0
    
    if prompt_tokens == 0 and completion_tokens == 0:
        return 0.0, 0, 0
    
    # Map model name to pricing key
    pricing_key = None
    if model_name in TOOL_PRICING:
        pricing_key = model_name
    elif MODEL_NAME and model_name == MODEL_NAME:  # Orchestrator
        # Try MODEL_TYPE first
        if MODEL_TYPE and MODEL_TYPE in TOOL_PRICING:
            pricing_key = MODEL_TYPE
        # If MODEL_TYPE not found, try to extract from MODEL_NAME
        elif MODEL_NAME:
            # Check if MODEL_NAME contains Nemotron
            if 'nemotron' in MODEL_NAME.lower() or 'orchestrator' in MODEL_NAME.lower():
                if 'Nemotron-Orchestrator-8B' in TOOL_PRICING:
                    pricing_key = 'Nemotron-Orchestrator-8B'
                # Fallback: try to match any part of the path/name
                else:
                    for key in TOOL_PRICING.keys():
                        if isinstance(key, str) and 'nemotron' in key.lower():
                            pricing_key = key
                            break
            # If MODEL_NAME itself is in TOOL_PRICING, use it
            elif MODEL_NAME in TOOL_PRICING:
                pricing_key = MODEL_NAME
    
    if pricing_key and pricing_key in TOOL_PRICING:
        pricing = TOOL_PRICING[pricing_key]
        # Pricing is per million tokens, so divide by 1,000,000
        cost = (prompt_tokens * pricing.get("input_tokens_per_million", 0) / 1000000 +
                completion_tokens * pricing.get("output_tokens_per_million", 0) / 1000000)
        return cost, prompt_tokens, completion_tokens
    
    return 0.0, prompt_tokens, completion_tokens

def call_tool(arguments):
    start_time = time.time()
    # Initialize cost tracking for this tool call
    tool_cost = 0.0
    tool_prompt_tokens = 0
    tool_completion_tokens = 0
    tool_prompt = None  # Initialize prompt storage
    tool_messages = None  # Initialize messages storage

    temperature = 1.0
    
    if arguments['tool']=='enhance_reasoning' or arguments['tool']=='code':
        supported_models = [MODEL_MAPPING[m] for m in ALL_TOOLS['enhance_reasoning']['model']]
        assert arguments['model'] in supported_models,f"Model {arguments['model']} is not supported in enhance_reasoning. Support models: {supported_models}"
        prompt = arguments['context_str'].strip()+'\n\n'
        prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write additional python code that will give intermidiate results after execution. Wrap the code within ```python and ```. The code should be self-contained with all the import and initialization."
        tool_prompt = prompt  # Store prompt
        model_name = arguments['model']
        response = ''
        if 'gpt-5' in model_name.lower() or 'claude' in model_name.lower():
            response = get_llm_response_with_retry(model=model_name,messages=prompt,return_raw_response=True,temperature=1,max_length=40000)
            # Track cost
            cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
            tool_cost += cost
            tool_prompt_tokens += prompt_toks
            tool_completion_tokens += completion_toks
        elif 'qwen2.5-coder' in model_name.lower() or 'nemotron' in model_name.lower() or '235' in model_name.lower():
            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=temperature,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            # Extract usage info from vLLM response (OpenAI-compatible ChatCompletion object)
            # vLLM/SGLang returns ChatCompletion with .usage attribute, same as API models
            if not isinstance(response, str) and hasattr(response, 'usage') and response.usage is not None:
                cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
                tool_cost += cost
                tool_prompt_tokens += prompt_toks
                tool_completion_tokens += completion_toks
            if isinstance(response, str):
                response = get_llm_response_with_retry(
                    model=model_name,
                    messages=prompt,
                    return_raw_response=True,
                    model_type="vllm",
                    max_length=8000,
                    temperature=temperature,
                    model_config=arguments["vllm_model_configs"][model_name],
                    model_config_path=arguments["vllm_model_configs"]["vllm_model_config_path"],
                    model_config_idx=arguments["eid"],
                )
                if not isinstance(response, str) and hasattr(response, "usage") and response.usage:
                    cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
                    tool_cost += cost
                    tool_prompt_tokens += prompt_toks
                    tool_completion_tokens += completion_toks
        elif 'qwen3-8b' in model_name.lower() or 'llama-3.3' in model_name.lower():
            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=temperature,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            # Extract usage info from vLLM response (OpenAI-compatible ChatCompletion object)
            # vLLM/SGLang returns ChatCompletion with .usage attribute, same as API models
            if not isinstance(response, str) and hasattr(response, 'usage') and response.usage is not None:
                cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
                tool_cost += cost
                tool_prompt_tokens += prompt_toks
                tool_completion_tokens += completion_toks
        if isinstance(response,str):
            arguments['generated_code'] = ''
            arguments['exec_result'] = ''
            arguments['response'] = ''  # Save raw response
            # Set cost fields even on early return
            arguments['_cost'] = tool_cost
            arguments['_prompt_tokens'] = tool_prompt_tokens
            arguments['_completion_tokens'] = tool_completion_tokens
            arguments['prompt_tokens'] = tool_prompt_tokens
            arguments['completion_tokens'] = tool_completion_tokens
            # Store prompt even on early return
            arguments['_prompt'] = tool_prompt
            arguments['_messages'] = tool_messages if tool_messages else None
            return arguments
        
        response_str = ''
        try:
            # Use the same extraction logic as extract_response_content_and_tool_calls
            # OpenAI format (GPT-5, vLLM)
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_str = response.choices[0].message.content or ""
            # Claude format (Anthropic Message object)
            elif hasattr(response, 'content') and not hasattr(response, 'choices'):
                content_parts = []
                for block in response.content:
                    if hasattr(block, 'text') and block.text:
                        content_parts.append(block.text)
                    elif isinstance(block, dict) and 'text' in block:
                        content_parts.append(block['text'])
                response_str = "\n".join(content_parts) if content_parts else ""
            # Fallback: try dict access for Claude
            elif isinstance(response, dict) and 'content' in response:
                if isinstance(response['content'], list) and len(response['content']) > 0:
                    if isinstance(response['content'][0], dict) and 'text' in response['content'][0]:
                        response_str = response['content'][0]['text'] or ""
        except Exception as e:
            print(f"[{arguments.get('id', 'unknown')}] WARNING: Failed to extract raw response for enhance_reasoning: {e}")
            response_str = ''
        
        # Extract generated code from response
        try:
            if 'claude' in model_name.lower():
                # Claude format - check both possible structures
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    content = response.choices[0].message.content
                elif hasattr(response, 'content') and len(response.content) > 0:
                    if hasattr(response.content[0], 'text'):
                        content = response.content[0].text
                    elif isinstance(response.content[0], dict) and 'text' in response.content[0]:
                        content = response['content'][0]['text']
                    else:
                        content = response_str
                else:
                    content = response_str
                generated_code = content.split('```python')[-1].split('```')[0] if content else ''
            else:
                # OpenAI/vLLM format
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    content = response.choices[0].message.content or ""
                elif isinstance(response, dict) and 'content' in response:
                    content = response['content'][0]['text'] if isinstance(response['content'], list) and len(response['content']) > 0 and 'text' in response['content'][0] else response_str
                else:
                    content = response_str
                generated_code = content.split('```python')[-1].split('```')[0] if content else ''
        except Exception as e:
            print(f"[{arguments.get('id', 'unknown')}] WARNING: Failed to extract generated_code: {e}")
            generated_code = ''
        if generated_code=='':
            arguments['generated_code'] = ''
            arguments['exec_result'] = ''
            arguments['response'] = response_str  # Save raw response even if code extraction failed
            # Set cost fields even on early return
            arguments['_cost'] = tool_cost
            arguments['_prompt_tokens'] = tool_prompt_tokens
            arguments['_completion_tokens'] = tool_completion_tokens
            arguments['prompt_tokens'] = tool_prompt_tokens
            arguments['completion_tokens'] = tool_completion_tokens
            # Store prompt even on early return
            arguments['_prompt'] = tool_prompt
            arguments['_messages'] = tool_messages if tool_messages else None
            return arguments
        code_path = str(os.path.join(arguments['cur_output_dir'],f'exec_code_{arguments["id"]}.py'))
        with open(code_path,'w') as f:
            f.write(generated_code)
        exec_result = ''
        exec_start = time.time()
        try:
            exec_result = subprocess.run(['python', code_path], timeout=60, capture_output=True, text=True)
            exec_time = time.time()-exec_start
            exec_result = exec_result.stdout
            with open(os.path.join(arguments['cur_output_dir'],f'exec_out_{arguments["id"]}.txt'),'w') as f:
                f.write(exec_result)
        except Exception as e:
            pass
        exec_time = time.time() - exec_start
        arguments['generated_code'] = generated_code
        arguments['exec_result'] = exec_result
        arguments['response'] = response_str  # Save raw response content
        arguments['_cost'] = tool_cost
        arguments['_prompt_tokens'] = tool_prompt_tokens
        arguments['_completion_tokens'] = tool_completion_tokens
        arguments['prompt_tokens'] = tool_prompt_tokens
        arguments['completion_tokens'] = tool_completion_tokens
        return arguments
    
    elif arguments['tool']=='answer':
        prompt = arguments['context_str'].strip()+'\n\n'+arguments['problem']
        model_name = arguments['model']
        
        # Check if context fits model's limit - FAIL if too long (don't fallback)
        # This lets the skill learner learn: "this model can't handle long context"
        max_context = MODEL_CONTEXT_LIMITS.get(model_name, 100000)
        # Use accurate tokenizer count instead of rough estimate
        estimated_tokens = estimate_context_tokens(arguments['context_str'], arguments['problem'])
        
        if estimated_tokens > max_context:
            # Record failure with detailed info for skill learning
            print(f"[{arguments.get('id', 'unknown')}] Context too long for {model_name}: ~{estimated_tokens} tokens > {max_context} limit - FAILING (learnable)")
            tool_prompt = prompt  # Store prompt before early return
            arguments['response'] = f'CONTEXT_TOO_LONG: {estimated_tokens} tokens exceeds {model_name} limit of {max_context}'
            arguments['pred'] = ''
            arguments['correctness'] = False
            arguments['_error'] = 'context_too_long'
            arguments['_estimated_tokens'] = estimated_tokens
            arguments['_model_limit'] = max_context
            arguments['_failure_skill'] = 'answer.long_context'  # Skill the model lacks
            # Store prompt even on failure
            arguments['_prompt'] = tool_prompt
            arguments['_messages'] = None
            return arguments
        
        response_str = ''
        pred = ''

        if 'qwen3' in arguments['model'].lower() and not '235' in arguments['model'].lower():
            model_name = arguments['model']
            tool_prompt = prompt  # Store prompt
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]
            tool_messages = messages  # Store messages
            arguments['messages'] = messages
            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=8000,temperature=temperature,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            # Track cost
            cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
            tool_cost += cost
            tool_prompt_tokens += prompt_toks
            tool_completion_tokens += completion_toks
            if isinstance(response,str):
                arguments['response'] = ''
                arguments['pred'] = ''
                arguments['correctness'] = False
                arguments['_cost'] = tool_cost
                arguments['_prompt_tokens'] = tool_prompt_tokens
                arguments['_completion_tokens'] = tool_completion_tokens
                return arguments
            response_str = response.choices[0].message.content
            if not isinstance(response_str,str) or not '\\boxed{' in response_str:
                pred = ''
            else:
                pred_components = response.choices[0].message.content.split('\\boxed{')[-1].split('}')[:-1]
                pred = '}'.join(pred_components).strip()
        elif 'qwen2.5-math' in arguments['model'].lower():
            model_name = arguments['model']
            tool_prompt = prompt  # Store prompt
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]
            tool_messages = messages  # Store messages
            arguments['messages'] = messages
            response = get_llm_response(model=model_name,messages=messages,return_raw_response=True,model_type='vllm',max_length=2000,temperature=temperature,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            # Track cost
            cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
            tool_cost += cost
            tool_prompt_tokens += prompt_toks
            tool_completion_tokens += completion_toks
            if isinstance(response,str):
                arguments['response'] = ''
                arguments['pred'] = ''
                arguments['correctness'] = False
                arguments['_cost'] = tool_cost
                arguments['_prompt_tokens'] = tool_prompt_tokens
                arguments['_completion_tokens'] = tool_completion_tokens
                return arguments
            response_str = response.choices[0].message.content
            if not isinstance(response_str,str) or not '\\boxed{' in response_str:
                pred = ''
            else:
                pred_components = response.choices[0].message.content.split('\\boxed{')[-1].split('}')[:-1]
                pred = '}'.join(pred_components).strip()
        elif 'gpt-5' in arguments['model'].lower() or 'claude' in arguments['model'].lower():
            model_name = arguments['model']
            prompt += ("\n\nTake a deep breath and think hard with high reasoning, wrap the thoughts within <think> and </think>, and wrap only the exact answer without any explanation within <answer> and </answer>."
                        "Output using the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>")
            tool_prompt = prompt  # Store final prompt (with instructions)
            tool_messages = None  # This branch uses prompt string, not messages list
            arguments['messages'] = prompt
            response = get_llm_response_with_retry(model=model_name,messages=prompt,return_raw_response=True,max_length=40000)
            # Track cost
            cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
            tool_cost += cost
            tool_prompt_tokens += prompt_toks
            tool_completion_tokens += completion_toks
            if isinstance(response,str):
                arguments['response'] = ''
                arguments['pred'] = ''
                arguments['correctness'] = False
                arguments['_cost'] = tool_cost
                arguments['_prompt_tokens'] = tool_prompt_tokens
                arguments['_completion_tokens'] = tool_completion_tokens
                return arguments
            response_str = response.choices[0].message.content
            if isinstance(response_str,str):
                pred = response_str.split('<answer>')[-1].split('</answer>')[0].strip()
            else:
                pred = ''
        elif 'llama-3.3' in arguments['model'].lower():
            model_name = arguments['model']
            prompt += "\nWrap the thinking process and explanation between <think> and </think> and wrap only the exact answer without any explanation within <answer> and </answer>."
            tool_prompt = prompt  # Store final prompt
            tool_messages = None  # This branch uses prompt string, not messages list
            arguments['messages'] = prompt
            response = get_llm_response(model=model_name,messages=prompt,return_raw_response=True,model_type='vllm',max_length=40000,temperature=temperature,model_config=arguments['vllm_model_configs'][model_name],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            # Track cost for successful vLLM response
            if not isinstance(response, str):
                cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
                tool_cost += cost
                tool_prompt_tokens += prompt_toks
                tool_completion_tokens += completion_toks
            if isinstance(response, str):
                response = get_llm_response_with_retry(
                    model=model_name,
                    messages=prompt,
                    return_raw_response=True,
                    model_type="vllm",
                    max_length=40000,
                    temperature=temperature,
                    model_config=arguments["vllm_model_configs"][model_name],
                    model_config_path=arguments["vllm_model_configs"]["vllm_model_config_path"],
                    model_config_idx=arguments["eid"],
                )
                if not isinstance(response, str) and hasattr(response, "usage") and response.usage:
                    cost, prompt_toks, completion_toks = calculate_cost(response, model_name)
                    tool_cost += cost
                    tool_prompt_tokens += prompt_toks
                    tool_completion_tokens += completion_toks
                if isinstance(response, str):
                    arguments['response'] = ''
                    arguments['pred'] = ''
                    arguments['correctness'] = False
                    # Store prompt even on error
                    arguments['_prompt'] = tool_prompt
                    arguments['_messages'] = tool_messages if tool_messages else None
                    return arguments
            response_str = response.choices[0].message.content
            if isinstance(response_str,str):
                pred = response.choices[0].message.content.split('<answer>')[-1].split('</answer>')[0].strip()
            else:
                pred = ''
        
        if pred.strip()=='' or len(pred.split(' '))>500:
            correctness = False
        elif pred.strip().lower()==arguments['answer'].strip().lower():
            correctness = True
        else:
            eval_prompt = (f"Question: {arguments['problem']}\n\n"
                        f"Student answer: {pred}\n\n"
                        f"Reference answer: {arguments['answer']}\n\n"
                        "Assume that the reference answer is correct. Output <correct>True</correct> if the student answer matches the reference answer. Output <correct>False</correct> if the student answer does not match the reference answer.")
            eval_response = get_llm_response_with_retry(model='gpt-5-mini',messages=eval_prompt,temperature=1)
            eval_result = eval_response.split('<correct>')[-1].split('</correct>')[0]
            if eval_result.lower()=='true':
                correctness = True
            else:
                correctness = False
        arguments['response'] = response_str
        arguments['pred'] = pred
        arguments['correctness'] = correctness
        arguments['_cost'] = tool_cost
        arguments['_prompt_tokens'] = tool_prompt_tokens
        arguments['_completion_tokens'] = tool_completion_tokens
        arguments['prompt_tokens'] = tool_prompt_tokens
        arguments['completion_tokens'] = tool_completion_tokens
        # Store prompt for this tool call
        arguments['_prompt'] = tool_prompt
        arguments['_messages'] = tool_messages if tool_messages else None
        return arguments

    elif arguments['tool']=='search':
        contents = []
        prompt = arguments['context_str'].strip()+'\n\n'
        # Use HLE-specific search prompt if IS_HLE is True
        if IS_HLE:
            prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please write a query to search for a piece of relevant and missing information. The query should be a few key words about the information to search or a short sentence. Wrap the query within <query> and </query>."
        else:
            prompt += f"Question: {arguments['problem']}\nInstead of directly answering the question, please think hard and write a concise query to search Wikipedia. Wrap the query within <query> and </query>."
        tool_prompt = prompt  # Store prompt
        cur_query_writer = arguments['model']
        query_to_call = None
        if 'gpt-5' in cur_query_writer.lower():
            response = get_llm_response_with_retry(model=cur_query_writer,messages=prompt,return_raw_response=True,temperature=1,max_length=40000)
            # Track cost
            cost, prompt_toks, completion_toks = calculate_cost(response, cur_query_writer)
            tool_cost += cost
            tool_prompt_tokens += prompt_toks
            tool_completion_tokens += completion_toks
            if isinstance(response,str) or not response.choices[0].message.content:
                query_to_call = arguments['problem']
            else:
                query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
        elif 'claude' in cur_query_writer.lower():
            response = get_llm_response_with_retry(model=cur_query_writer,messages=prompt,return_raw_response=True,temperature=1,max_length=40000)
            # Track cost
            cost, prompt_toks, completion_toks = calculate_cost(response, cur_query_writer)
            tool_cost += cost
            tool_prompt_tokens += prompt_toks
            tool_completion_tokens += completion_toks
            if isinstance(response,str) or not response['content'][0]['text']:
                query_to_call = arguments['problem']
            else:
                query_to_call = response['content'][0]['text'].split('<query>')[-1].split('</query>')[0]
        elif 'qwen3' in cur_query_writer.lower():
            response = get_llm_response(model=cur_query_writer,messages=prompt,return_raw_response=True,model_type='vllm',max_length=8000,temperature=temperature,model_config=arguments['vllm_model_configs'][cur_query_writer],model_config_path=arguments['vllm_model_configs']['vllm_model_config_path'],model_config_idx=arguments['eid'])
            # Track cost
            cost, prompt_toks, completion_toks = calculate_cost(response, cur_query_writer)
            tool_cost += cost
            tool_prompt_tokens += prompt_toks
            tool_completion_tokens += completion_toks
            if isinstance(response,str):
                query_to_call = arguments['problem']
            else:
                query_to_call = response.choices[0].message.content.split('<query>')[-1].split('</query>')[0]
        # Use HLE-specific query length check if IS_HLE is True
        if IS_HLE:
            query_min_len = 5
        else:
            query_min_len = 10
        if query_to_call is None or len(query_to_call)<query_min_len:
            pass
        else:
            query_length = len(tokenizer(query_to_call)['input_ids'])
            assert len(query_to_call)>5,f"{query_to_call}"
            # Use HLE-specific topk if IS_HLE is True
            topk = 50 if IS_HLE else 150
            payload = {
                "queries": [query_to_call[:390]],
                "topk": topk,
                "return_scores": True,
                "eid": arguments['id']
            }
            # Add cache directory if specified
            if RETRIEVER_CACHE_DIR:
                payload["new_cache_dir"] = RETRIEVER_CACHE_DIR
            results = None
            all_vllm_model_configs = arguments['vllm_model_configs']
            while not results:
                try:
                    # Use HLE-specific retrieval config (only 'retrieval', not 'wiki_retrieval')
                    if IS_HLE:
                        cur_model_config = random.choice(all_vllm_model_configs['retrieval'])
                    else:
                        # FRAMES: prefer wiki_retrieval, fallback to retrieval
                        if 'wiki_retrieval' in all_vllm_model_configs:
                            cur_model_config = random.choice(all_vllm_model_configs['wiki_retrieval'])
                        else:
                            cur_model_config = random.choice(all_vllm_model_configs['retrieval'])
                    results = requests.post(f'http://{cur_model_config["ip_addr"]}:{cur_model_config["port"]}/retrieve', json=payload).json()
                except Exception as search_error:
                    time.sleep(3)
            for r in results[0]:
                if 'content' in r['document']:
                    contents.append(r['document']['content'])
                elif 'contents' in r['document']:
                    contents.append(r['document']['contents'])
        arguments['search_results_data'] = contents
        if 'tokenizer' in arguments:
            arguments.pop('tokenizer')
        arguments['_cost'] = tool_cost
        arguments['_prompt_tokens'] = tool_prompt_tokens
        arguments['_completion_tokens'] = tool_completion_tokens
        arguments['prompt_tokens'] = tool_prompt_tokens
        arguments['completion_tokens'] = tool_completion_tokens
        # Store prompt for this tool call
        arguments['_prompt'] = tool_prompt
        arguments['_messages'] = tool_messages if tool_messages else None
        return arguments

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed as futures_as_completed
from typing import Iterable, Tuple, Any, Callable

# Synchronous version for nested calls (when already in a thread executor)
def run_all_sync(
    task_list: Iterable[Tuple[Callable[[Any], Any], Any]],
    concurrency: int = 2,
    progress: bool = False,
    return_exceptions: bool = False,
):
    """Synchronous version of run_all for use when already in a thread executor."""
    task_list = list(task_list)
    results = [None] * len(task_list)
    
    if progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(task_list), desc="Processing tools", unit="tool")
    else:
        pbar = None
    
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(func, arg): idx 
                for idx, (func, arg) in enumerate(task_list)
            }
            
            # Process completed tasks
            for future in futures_as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results[idx] = res
                    if pbar:
                        pbar.update(1)
                except Exception as exc:
                    if return_exceptions:
                        results[idx] = exc
                        if pbar:
                            pbar.update(1)
                    else:
                        # Cancel remaining tasks and re-raise
                        for f in future_to_idx:
                            f.cancel()
                        if pbar:
                            pbar.close()
                        raise exc
    finally:
        if pbar:
            pbar.close()
    
    return results

# task_list is an iterable of (func, arg) pairs
async def run_all(
    task_list: Iterable[Tuple[Callable[[Any], Any], Any]],
    concurrency: int = 2,
    progress: bool = False,
    return_exceptions: bool = False,
):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, get the current event loop
        loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(concurrency)

    # create the executor sized to your concurrency gate
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # wrap each task so it obeys the semaphore
        async def run_one(idx: int, func: Callable, arg: Any):
            async with sem:
                # try:
                if asyncio.iscoroutinefunction(func):
                    res = await func(arg)
                else:
                    res = await loop.run_in_executor(executor, func, arg)
                return idx, res, None

        task_list = list(task_list)
        tasks = [asyncio.create_task(run_one(i, f, a))
                 for i, (f, a) in enumerate(task_list)]

        results = [None] * len(tasks)

        if progress:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks), desc="Processing samples", unit="sample")
        else:
            pbar = None

        try:
            # update progress as tasks complete
            for fut in asyncio.as_completed(tasks):
                idx, res, err = await fut
                if err is None:
                    results[idx] = res
                    if pbar:
                        # Try to get sample ID from result for better progress display
                        sample_id = "unknown"
                        if isinstance(res, dict) and 'id' in res:
                            sample_id = res['id']
                        elif isinstance(res, dict) and 'all_tool_calls' in res:
                            # Try to extract ID from the example if available
                            pass
                        pbar.set_postfix({"current": sample_id})
                        pbar.update(1)
                else:
                    if return_exceptions:
                        results[idx] = err
                        if pbar:
                            pbar.update(1)
                    else:
                        # cancel remaining, then re-raise the first error
                        for t in tasks:
                            t.cancel()
                        with contextlib.suppress(Exception):
                            await asyncio.gather(*tasks, return_exceptions=True)
                        raise err
        finally:
            if pbar:
                pbar.close()

        return results

# =============================================================================
# Checkpoint Functions for Efficient Multi-Model Exploration
# =============================================================================
def save_checkpoint(problem_id, checkpoint_dir, state):
    """Save checkpoint state for a problem after each stage."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{problem_id}_checkpoint.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(state, f, indent=2)

def load_checkpoint(problem_id, checkpoint_dir):
    """Load checkpoint state for a problem."""
    checkpoint_file = os.path.join(checkpoint_dir, f"{problem_id}_checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def run_single(e):
    doc_list = []
    code_list = []
    attempt_list = []
    exp_start_time = time.time()
    problem = e['question']
    user_problem = problem
    answer = e['answer']
    all_tool_calls = []
    final_correct = False
    final_answer_model = None
    final_pred = ''
    all_tool_responses = {}
    all_message_responses = {}
    used_tools = []
    start_step = 0
    reasoning_ever_used = False  # Track if enhance_reasoning was ever called
    search_ever_used = False  # Track if search was ever called
    
    # Initialize cost tracking
    total_cost_routed_all_tokens = 0.0  # All tokens from routed models
    total_cost_all_models_all_tokens = 0.0  # All tokens from routed + orchestrator
    total_cost_routed_completion_only = 0.0  # Only completion tokens from routed models
    total_cost_all_models_completion_only = 0.0  # Only completion tokens from routed + orchestrator
    orchestrator_cost = 0.0  # Total cost from orchestrator (all tokens)
    orchestrator_prompt_tokens = 0
    orchestrator_completion_tokens = 0
    
    # Load checkpoint if specified
    if LOAD_CHECKPOINT_DIR:
        checkpoint = load_checkpoint(e['id'], LOAD_CHECKPOINT_DIR)
        if checkpoint:
            doc_list = checkpoint.get('doc_list', [])
            code_list = checkpoint.get('code_list', [])
            attempt_list = checkpoint.get('attempt_list', [])
            all_tool_calls = checkpoint.get('all_tool_calls', [])
            all_tool_responses = checkpoint.get('all_tool_responses', {})
            all_message_responses = checkpoint.get('all_message_responses', {})
            used_tools = checkpoint.get('used_tools', [])
            start_step = checkpoint.get('step', 0)
            total_cost_routed_all_tokens = checkpoint.get('total_cost_routed_all_tokens', 0.0)
            total_cost_all_models_all_tokens = checkpoint.get('total_cost_all_models_all_tokens', 0.0)
            total_cost_routed_completion_only = checkpoint.get('total_cost_routed_completion_only', 0.0)
            total_cost_all_models_completion_only = checkpoint.get('total_cost_all_models_completion_only', 0.0)
            orchestrator_cost = checkpoint.get('orchestrator_cost', 0.0)
            orchestrator_prompt_tokens = checkpoint.get('orchestrator_prompt_tokens', 0)
            orchestrator_completion_tokens = checkpoint.get('orchestrator_completion_tokens', 0)
            print(f"  Loaded checkpoint for {e['id']}: step={start_step}, docs={len(doc_list)}, code={len(code_list)}")
    
    finish_flag = False  # Track if we've answered
    for step in range(start_step, MAX_ROUNDS):
        cur_output_dir = os.path.join(my_output_dir,f"step_{step}")
        if not os.path.isdir(os.path.join(cur_output_dir,'tool_return')):
            try:
                os.makedirs(os.path.join(cur_output_dir,'tool_return'))
            except:
                pass
        tools = []
        # Always include all tools - the orchestrator should be able to search again if needed
        for t in raw_tools:
            tools.append(t)
        doc_str = ''
        # Use HLE-specific doc truncation if IS_HLE is True
        doc_truncate_len = 1200 if IS_HLE else 4000
        for doc_idx, doc in enumerate(doc_list):
            doc_str += f"Doc {doc_idx+1}: {doc[:doc_truncate_len]}{' ...' if IS_HLE and len(doc) > doc_truncate_len else ''}\n\n"
        code_str = ''
        for code_idx, code_piece in enumerate(code_list):
            code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
        attempt_str = ''
        for attempt_idx, attempt in enumerate(attempt_list):
            not_useful_note = " (NOT USEFUL - asks questions or defers)" if attempt.get('not_useful', False) else ""
            attempt_str += f"Attempt{attempt_idx+1} answer by {attempt['model']}: {attempt['answer']}{not_useful_note}\n"
        str_cut = cut_seq(seq=attempt_str,l=8000)
        attempt_str = str_cut['string_after_cut']
        if not attempt_str.startswith('Attempt') and len(attempt_str)>0:
            attempt_str = 'Attempt answer: '+attempt_str
        str_cut = cut_seq(seq=code_str+attempt_str,l=12000)
        code_attempt_str = str_cut['string_after_cut']
        code_attempt_str_len = str_cut['effective_length']
        if not code_attempt_str.startswith('```') and len(code_attempt_str)>0:
            code_attempt_str = '```\n'+code_attempt_str
        doc_flag = False
        # Use HLE-specific context limits if IS_HLE is True
        if IS_HLE:
            problem_length = len(tokenizer(problem)['input_ids'])
            context_limit = 27000 - problem_length
            if code_attempt_str_len < context_limit:
                if code_attempt_str:
                    context_str = cut_seq(seq=doc_str+"\npython code and execution outputs:\n"+code_attempt_str,l=context_limit)
                else:
                    context_str = cut_seq(seq=doc_str,l=context_limit)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    doc_flag = True
                    context_str = 'Documents:\n'+context_str
            else:
                context_str = code_attempt_str
        else:
            # FRAMES: original logic
            if code_attempt_str_len<24000:
                context_str = cut_seq(seq=doc_str+"\npython code and execution outputs:\n"+code_attempt_str,l=24000)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    doc_flag = True
                    context_str = 'Documents:\n'+context_str
            else:
                context_str = code_attempt_str

        # Filter tools by context length - remove models that can't handle the context
        # Only filter when routing is enabled (orig didn't have this filtering)
        # NOTE: For answer tool, we skip filtering here because:
        # 1. The actual context will be truncated per model anyway
        # 2. We don't know which tool will be selected yet
        # 3. Filtering will happen later when we know the tool and actual context
        # For search/reasoning tools, filter based on orchestrator context
        if ROUTING_STRATEGY != "none" and HANDBOOK is not None:
            tools = filter_models_by_context_length(tools, context_str, problem, skip_answer_tool=True)

        removed_tool = None
        if len(used_tools)>1 and used_tools[-1]==used_tools[-2]:
            updated_tools = []
            removed_tool = used_tools[-1]
            for t in tools:
                if t['function']['name']!=used_tools[-1]:
                    updated_tools.append(t)
        else:
            updated_tools = tools
        
        is_last_step = (step == MAX_ROUNDS - 1)
        if is_last_step and final_pred == '':
            answer_tools = [t for t in tools if t['function']['name'] == 'answer']
            if answer_tools:
                updated_tools = answer_tools
                print(f"[{e['id']}] FORCE ANSWER: Reached max turns ({MAX_ROUNDS}), forcing answer tool")
            else:
                print(f"[{e['id']}] WARNING: On last step but answer tool not available!")
        
        cur_tool_set = [t['function']['name'] for t in updated_tools]
        
        if ROUTING_STRATEGY == "none" or HANDBOOK is None:
            orchestrator_max_length = 8000
            orchestrator_model_limit = MODEL_CONTEXT_LIMITS.get(MODEL_NAME, MODEL_CONTEXT_LIMITS.get(MODEL_TYPE, 100000))
            
            user_content = f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool."
            # For API models (GPT-5, Claude, Gemini), add explicit instructions to use function calling
            if ('gpt-5' in MODEL_NAME.lower() or 
                'claude' in MODEL_NAME.lower() or 
                'gemini' in MODEL_NAME.lower() or
                (MODEL_TYPE and ('gpt' in MODEL_TYPE.lower() or 'claude' in MODEL_TYPE.lower() or 'gemini' in MODEL_TYPE.lower()))):
                system_content = """You are a helpful assistant that uses function calling to solve problems. 

IMPORTANT: You have been provided with tools via the function calling API. These tools are available to you through the API's function calling mechanism - you MUST use them by calling the functions, not by describing what you would do.

Available tools (provided via function calling API):
- search: requires model parameter: search-1, search-2, or search-3
- enhance_reasoning: requires model parameter: reasoner-1, reasoner-2, or reasoner-3
- answer: requires model parameter: answer-1, answer-2, answer-3, answer-4, answer-math-1, or answer-math-2)

You must call one of these tools with the appropriate model parameter. The tools are available to you through the function calling API - use them!"""
            else:
                system_content = "You are good at using tools."
            
            # Check if prompt exceeds model limit and truncate/reduce if needed
            system_tokens = len(tokenizer(system_content)['input_ids'])
            user_tokens = len(tokenizer(user_content)['input_ids'])
            total_estimated_tokens = system_tokens + user_tokens + orchestrator_max_length
            
            if total_estimated_tokens > orchestrator_model_limit:
                print(f"[{e['id']}] Prompt exceeds model limit: ~{total_estimated_tokens} tokens > {orchestrator_model_limit} limit")
                # Try reducing max_length first (less aggressive than truncating context)
                new_max_length = max(1000, orchestrator_model_limit - system_tokens - user_tokens - 500) # 500 tokenbuffer
                if new_max_length < orchestrator_max_length:
                    orchestrator_max_length = new_max_length
                    print(f"[{e['id']}] Reduced max_length from 12000 to {orchestrator_max_length} to fit within model limit")
                else:
                    # If reducing max_length isn't enough, truncate context as last resort
                    print(f"[{e['id']}] Truncating context to fit within model limit...")
                    available_tokens = orchestrator_model_limit - system_tokens - orchestrator_max_length - 500
                    if user_content.startswith("Problem: "):
                        parts = user_content.split("\n\n", 2)
                        if len(parts) >= 3:
                            problem_part = parts[0]
                            context_part = parts[1]
                            ending = parts[2] if len(parts) > 2 else "Choose an appropriate tool."
                            problem_tokens = len(tokenizer(problem_part)['input_ids'])
                            ending_tokens = len(tokenizer(ending)['input_ids'])
                            context_available = available_tokens - problem_tokens - ending_tokens
                            if context_available > 0:
                                original_context_tokens = len(tokenizer(context_part)['input_ids'])
                                context_truncated = cut_seq(seq=context_part, l=context_available)
                                context_part = context_truncated['string_after_cut']
                                truncated_context_tokens = len(tokenizer(context_part)['input_ids'])
                                user_content = f"{problem_part}\n\n{context_part}\n\n{ending}"
                                print(f"[{e['id']}] Truncated context from ~{original_context_tokens} to ~{truncated_context_tokens} tokens")
            
            # Debug: Show system prompt for API models
            if ('gpt-5' in MODEL_NAME.lower() or 
                'claude' in MODEL_NAME.lower() or 
                'gemini' in MODEL_NAME.lower() or
                (MODEL_TYPE and ('gpt' in MODEL_TYPE.lower() or 'claude' in MODEL_TYPE.lower() or 'gemini' in MODEL_TYPE.lower()))):
                print(f"[{e['id']}] [DEBUG] System prompt for API model ({MODEL_NAME}): {system_content[:200]}...")
            
            chat = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            # API-based orchestrators (gpt-5, claude, gemini) use get_llm_response_with_retry
            # vLLM-based orchestrators use vllm_model_configs
            if ('gpt-5' in MODEL_NAME.lower() or 
                'claude' in MODEL_NAME.lower() or 
                'gemini' in MODEL_NAME.lower() or
                (MODEL_TYPE and ('gpt' in MODEL_TYPE.lower() or 'claude' in MODEL_TYPE.lower() or 'gemini' in MODEL_TYPE.lower()))):
                print(f"[{e['id']}] [DEBUG] Calling API orchestrator: {MODEL_NAME}, tools provided: {len(tools) if tools else 0}")
                if tools:
                    print(f"[{e['id']}] [DEBUG] Tool names: {[t.get('function', {}).get('name', 'unknown') for t in tools]}")
                response = get_llm_response_with_retry(
                    model=MODEL_NAME,
                    messages=chat,
                    return_raw_response=True,
                    temperature=1,
                    max_length=orchestrator_max_length,
                    tools=tools
                )
            else:
                model_config_key = None
                if MODEL_NAME in vllm_model_configs:
                    model_config_key = MODEL_NAME
                else:
                    for key in vllm_model_configs.keys():
                        if key == MODEL_NAME or key.endswith(MODEL_NAME) or MODEL_NAME in key:
                            model_config_key = key
                            break
                
                if model_config_key is None:
                    raise KeyError(f"Could not find model config for '{MODEL_NAME}' in vllm_model_configs. Available keys: {list(vllm_model_configs.keys())}")
                
                response = get_llm_response(
                    model=MODEL_NAME,
                    messages=chat,
                    return_raw_response=True,
                    model_type='vllm',
                    model_config=vllm_model_configs[model_config_key],
                    temperature=1,
                    max_length=orchestrator_max_length,
                    tools=tools,
                    model_config_path=vllm_model_configs['vllm_model_config_path'],
                    model_config_idx=e['eid']
                )
            
            if isinstance(response,str):
                print(f"[{e['id']}] [DEBUG] Response is a string, skipping")
                continue
            
            # Debug: Show response type and structure
            print(f"[{e['id']}] [DEBUG] Response type: {type(response)}")
            print(f"[{e['id']}] [DEBUG] Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')][:15]}")
            
            # Track orchestrator cost
            orch_cost, orch_prompt, orch_completion = calculate_cost(response, MODEL_NAME)
            orchestrator_cost += orch_cost
            orchestrator_prompt_tokens += orch_prompt
            orchestrator_completion_tokens += orch_completion
            total_cost_all_models_all_tokens += orch_cost
            total_cost_all_models_completion_only += (orch_completion * TOOL_PRICING.get(MODEL_TYPE, {}).get("output_tokens_per_million", 0) / 1000000 if MODEL_TYPE in TOOL_PRICING else 0)
            
            print(f"[{e['id']}] [DEBUG] Extracting content and tool calls from response...")
            # Extract content and tool calls - handle different response formats
            orchestrator_content, tool_calls = extract_response_content_and_tool_calls(response, MODEL_NAME)
            print(f"[{e['id']}] [DEBUG] Extraction result: content_length={len(orchestrator_content) if orchestrator_content else 0}, tool_calls={tool_calls}")
            
            # Initialize variables needed for tool processing
            cache_tool_calls = []
            if tool_calls is not None:
                print(f"[{e['id']}] [DEBUG] Processing {len(tool_calls)} tool call(s) from orchestrator")
                for idx, one_tool_call in enumerate(tool_calls):
                    try:
                        tool_name = one_tool_call.function.name
                        raw_args = one_tool_call.function.arguments
                        print(f"[{e['id']}] [DEBUG]   Tool call {idx+1}: name={tool_name}, raw_arguments={str(raw_args)[:200]}...")
                        # Parse arguments - could be JSON string or already a dict
                        if isinstance(one_tool_call.function.arguments, str):
                            tool_arguments = json.loads(one_tool_call.function.arguments)
                        elif isinstance(one_tool_call.function.arguments, dict):
                            tool_arguments = one_tool_call.function.arguments
                        else:
                            tool_arguments = {}
                        print(f"[{e['id']}] [DEBUG]   Tool call {idx+1}: parsed_arguments={tool_arguments}")
                        cache_tool_calls.append({
                            'tool_name': tool_name,
                            'tool_arguments': tool_arguments
                        })
                    except Exception as parse_error:
                        print(f"[{e['id']}] WARNING: Failed to parse tool call: {parse_error}")
                        import traceback
                        traceback.print_exc()
                        continue
                print(f"[{e['id']}] [DEBUG] Successfully parsed {len(cache_tool_calls)} tool call(s)")
            else:
                print(f"[{e['id']}] [DEBUG] No tool calls found in orchestrator response")
            routing_info_per_tool = {}  # Empty when routing disabled
            message_dict = {
                'content': orchestrator_content,
                'tool_calls': cache_tool_calls,
                'prompt_tokens': orch_prompt,
                'completion_tokens': orch_completion,
                'model': MODEL_NAME,
                'system_prompt': "You are good at using tools.",
                'user_prompt': f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool.",
            }
            
        else:
            # Routing enabled - use enhanced logic with handbook
            orchestrator_max_length = 4000
            orchestrator_model_limit = MODEL_CONTEXT_LIMITS.get(MODEL_NAME, MODEL_CONTEXT_LIMITS.get(MODEL_TYPE, 100000))
            
            # Enhanced prompt with skill handbook
            context_str_truncated = context_str
            if ROUTING_STRATEGY != "none" and HANDBOOK is not None:
                # Build prompt skeleton (with empty context) to get actual handbook + problem + instructions token count
                try:
                    from skillorchestra.prompts import build_skill_orchestrator_prompt
                    prompt_skeleton = build_skill_orchestrator_prompt(
                        problem=problem,
                        context_str="",
                        strategy=ROUTING_STRATEGY,
                        handbook=HANDBOOK,
                    )
                    prompt_without_context_tokens = len(tokenizer(prompt_skeleton)['input_ids'])
                except Exception:
                    prompt_without_context_tokens = 4000 + len(tokenizer(problem)['input_ids'])  # fallback
                system_tokens_estimate = len(tokenizer("You are a skill-based orchestrator for multi-step question answering.")['input_ids'])
                buffer_tokens = 2000
                context_available_estimate = orchestrator_model_limit - system_tokens_estimate - prompt_without_context_tokens - orchestrator_max_length - buffer_tokens
                # When handbook+problem exceeds limit, context_available can be negative - still truncate to fit
                context_available_estimate = max(500, context_available_estimate)
                
                if len(context_str) > 0:
                    context_tokens = len(tokenizer(context_str)['input_ids'])
                    if context_tokens > context_available_estimate:
                        print(f"[{e['id']}] Routing enabled: truncating context from ~{context_tokens} to ~{context_available_estimate} tokens "
                              f"(orchestrator_limit={orchestrator_model_limit}, handbook+problem=~{prompt_without_context_tokens} tokens)")
                        
                        # Parse context_str to identify document and code sections
                        doc_section = ""
                        code_section = ""
                        doc_header = "Documents:\n"
                        code_header = "\npython code and execution outputs:\n"
                        
                        if context_str.startswith(doc_header):
                            header_end = len(doc_header)
                            code_header_idx = context_str.find(code_header, header_end)
                            if code_header_idx != -1:
                                doc_section = context_str[header_end:code_header_idx]
                                code_section = context_str[code_header_idx + len(code_header):]
                            else:
                                doc_section = context_str[header_end:]
                        else:
                            code_section = context_str
                        
                        code_tokens = len(tokenizer(code_section)['input_ids']) if code_section else 0
                        
                        if code_tokens > context_available_estimate:
                            code_truncated = cut_seq(seq=code_section, l=context_available_estimate)['string_after_cut']
                            context_str_truncated = code_truncated
                            print(f"[{e['id']}]   Truncated code: {code_tokens} -> {len(tokenizer(code_truncated)['input_ids'])} tokens (no room for documents)")
                        else:
                            doc_available = context_available_estimate - code_tokens
                            if doc_available > 0 and doc_section:
                                combined = doc_header + doc_section + code_header + code_section
                                context_str_truncated = cut_seq(seq=combined, l=context_available_estimate)['string_after_cut']
                                print(f"[{e['id']}]   Added documents, truncated to fit: ~{len(tokenizer(context_str_truncated)['input_ids'])} tokens")
                            else:
                                context_str_truncated = code_section
                                print(f"[{e['id']}]   No room for documents, kept code only")
        
        # Build prompt based on routing strategy
        if step == 0:  # Debug logging on first step
            print(f"[{e['id']}] Prompt building check: ROUTING_STRATEGY={ROUTING_STRATEGY}, HANDBOOK={'loaded' if HANDBOOK is not None else 'None'}", flush=True)
        
        if ROUTING_STRATEGY != "none" and HANDBOOK is not None:
            try:
                from skillorchestra.prompts import build_skill_orchestrator_prompt
                if step == 0:  # Only print on first step to avoid spam
                    print(f"[{e['id']}] Building enhanced skill orchestrator prompt with handbook (strategy={ROUTING_STRATEGY})", flush=True)
                user_content = build_skill_orchestrator_prompt(
                    problem=problem,
                    context_str=context_str_truncated,  # Use truncated context_str (documents/code only)
                    strategy=ROUTING_STRATEGY,
                    handbook=HANDBOOK,
                )
                # For API models, add explicit function calling instructions to the system prompt
                if ('gpt-5' in MODEL_NAME.lower() or 
                    'claude' in MODEL_NAME.lower() or 
                    'gemini' in MODEL_NAME.lower() or
                    (MODEL_TYPE and ('gpt' in MODEL_TYPE.lower() or 'claude' in MODEL_TYPE.lower() or 'gemini' in MODEL_TYPE.lower()))):
                    system_content = """You are a skill-based orchestrator for multi-step question answering.

IMPORTANT: You have been provided with tools via the function calling API. These tools are available to you through the API's function calling mechanism - you MUST use them by calling the functions, not by describing what you would do.

Available tools (provided via function calling API):
- search: requires model parameter: search-1, search-2, or search-3
- enhance_reasoning: requires model parameter: reasoner-1, reasoner-2, or reasoner-3
- answer: requires model parameter: answer-1, answer-2, answer-3, answer-4, answer-math-1, or answer-math-2)

You must call one of these tools with the appropriate model parameter. The tools are available to you through the function calling API - use them!"""
                else:
                    system_content = "You are a skill-based orchestrator for multi-step question answering."
                # Debug: Print first 500 chars of prompt to verify handbook is included
                if step == 0:  # Only print on first step to avoid spam
                    print(f"[Router Debug] Enhanced prompt built successfully (strategy={ROUTING_STRATEGY})", flush=True)
                    print(f"[Router Debug] System prompt: {system_content}", flush=True)
                    print(f"[Router Debug] User prompt preview (first 500 chars): {user_content[:500]}...", flush=True)
            except Exception as exc:
                print(f"[{e['id']}] ERROR: Failed to build enhanced prompt: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                # Fallback to default only if enhanced prompt building fails (shouldn't happen)
                user_content = f"Problem: {problem}\n\n{context_str_truncated}\n\nChoose an appropriate tool."
                # For API models, add explicit function calling instructions
                if ('gpt-5' in MODEL_NAME.lower() or 
                    'claude' in MODEL_NAME.lower() or 
                    'gemini' in MODEL_NAME.lower() or
                    (MODEL_TYPE and ('gpt' in MODEL_TYPE.lower() or 'claude' in MODEL_TYPE.lower() or 'gemini' in MODEL_TYPE.lower()))):
                    system_content = """You are a helpful assistant that uses function calling to solve problems. 

IMPORTANT: You have been provided with tools via the function calling API. These tools are available to you through the API's function calling mechanism - you MUST use them by calling the functions, not by describing what you would do.

Available tools (provided via function calling API):
- search: Find missing information (requires model parameter: search-1, search-2, or search-3)
- enhance_reasoning: Write and execute Python code (requires model parameter: reasoner-1, reasoner-2, or reasoner-3)
- answer: Provide the final answer (requires model parameter: answer-1, answer-2, answer-3, answer-4, answer-math-1, or answer-math-2)

Example tool call format:
- To search: call function "search" with arguments {"model": "search-1"}
- To enhance reasoning: call function "enhance_reasoning" with arguments {"model": "reasoner-1"}
- To answer: call function "answer" with arguments {"model": "answer-1"}

You must call one of these tools with the appropriate model parameter. The tools are available to you through the function calling API - use them!"""
                else:
                    system_content = "You are good at using tools."
        else:
            if step == 0:  # Only print on first step to avoid spam
                print(f"[{e['id']}] Using default prompt (strategy={ROUTING_STRATEGY}, handbook={'loaded' if HANDBOOK is not None else 'None'})", flush=True)
            # Default prompt (only used if routing is disabled)
            if step == 0 and ROUTING_STRATEGY != "none":
                print(f"[Router Debug] WARNING: Routing strategy is '{ROUTING_STRATEGY}' but handbook is None!", flush=True)
            # When routing is disabled, use context_str (not truncated) to match orig behavior
            user_content = f"Problem: {problem}\n\n{context_str}\n\nChoose an appropriate tool."
            # For API models, add explicit function calling instructions
            if ('gpt-5' in MODEL_NAME.lower() or 
                'claude' in MODEL_NAME.lower() or 
                'gemini' in MODEL_NAME.lower() or
                (MODEL_TYPE and ('gpt' in MODEL_TYPE.lower() or 'claude' in MODEL_TYPE.lower() or 'gemini' in MODEL_TYPE.lower()))):
                system_content = """You are a helpful assistant that uses function calling to solve problems. 
You have access to tools (functions) that you MUST call to solve the problem. 
When you need to use a tool, you MUST call it using the function calling format. 
Do not just describe what you would do - actually call the function/tool.
Available tools: search, enhance_reasoning, answer.
You must call one of these tools with the appropriate model parameter."""
            else:
                system_content = "You are good at using tools."
        
        # Final safety check: verify prompt fits within limit
        # Always check to prevent exceeding model's context length (even when routing disabled)
        # Get orchestrator model limit (should be set in both routing enabled/disabled paths)
        if 'orchestrator_model_limit' not in locals():
            orchestrator_model_limit = MODEL_CONTEXT_LIMITS.get(MODEL_NAME, MODEL_CONTEXT_LIMITS.get(MODEL_TYPE, 100000))
        
        system_tokens = len(tokenizer(system_content)['input_ids'])
        user_tokens = len(tokenizer(user_content)['input_ids'])
        total_estimated_tokens = system_tokens + user_tokens + orchestrator_max_length
        
        if total_estimated_tokens > orchestrator_model_limit:
            print(f"[{e['id']}] Prompt exceeds model limit: ~{total_estimated_tokens} tokens > {orchestrator_model_limit} limit")
            
            # Only truncate if routing is enabled (to preserve handbook sections)
            # If routing disabled, reduce max_length instead (to match orig behavior)
            if ROUTING_STRATEGY != "none" and HANDBOOK is not None:
                # Enhanced prompt with handbook - truncate context section
                print(f"[{e['id']}] Truncating context section in enhanced prompt (preserving handbook data)...")
                print(f"[{e['id']}] Prompt still too long after context_str truncation: ~{total_estimated_tokens} tokens > {orchestrator_model_limit} limit")
                print(f"[{e['id']}] Truncating context section in enhanced prompt (preserving handbook data)...")
                
                # Calculate available tokens
                available_tokens = orchestrator_model_limit - system_tokens - orchestrator_max_length - 1000  # 1000 token buffer
                
                if available_tokens > 0:
                    # For enhanced prompts, find and truncate ONLY the context section (preserve handbook sections)
                    if "## Current Context" in user_content:
                        # Enhanced prompt - find and truncate context section only (preserve handbook sections)
                        parts = user_content.split("## Current Context", 1)
                        if len(parts) == 2:
                            before_context = parts[0]  # Problem + header + handbook sections (PRESERVE THIS)
                            after_context = parts[1]  # Context section + everything after
                            
                            # Find where context section ends (look for "---" separator or next section)
                            context_end_idx = after_context.find("\n---")
                            if context_end_idx == -1:
                                # No separator, look for next section header
                                next_section_idx = after_context.find("\n##")
                                if next_section_idx != -1:
                                    context_end_idx = next_section_idx
                                else:
                                    context_end_idx = len(after_context)
                            
                            context_section = after_context[:context_end_idx]
                            after_context_section = after_context[context_end_idx:] if context_end_idx < len(after_context) else ""
                            
                            # Calculate tokens for non-context parts (preserve handbook data)
                            before_tokens = len(tokenizer(before_context)['input_ids'])
                            after_tokens = len(tokenizer(after_context_section)['input_ids']) if after_context_section else 0
                            context_available = available_tokens - before_tokens - after_tokens
                            
                            if context_available > 0:
                                # Progressive truncation: truncate documents and code in context section
                                context_truncated = cut_seq(seq=context_section, l=context_available)
                                context_section = context_truncated['string_after_cut']
                                
                                # Rebuild user_content (preserving handbook sections)
                                user_content = before_context + "## Current Context" + context_section + after_context_section
                                
                                # Re-estimate
                                user_tokens = len(tokenizer(user_content)['input_ids'])
                                total_estimated_tokens = system_tokens + user_tokens + orchestrator_max_length
                                print(f"[{e['id']}] After final truncation: ~{total_estimated_tokens} tokens (limit: {orchestrator_model_limit})")
                                
                                # If still exceeds limit, reduce max_length
                                if total_estimated_tokens > orchestrator_model_limit:
                                    # Reduce max_length to fit within limit
                                    new_max_length = max(1000, orchestrator_model_limit - system_tokens - user_tokens - 500)  # 500 token buffer
                                    if new_max_length < orchestrator_max_length:
                                        orchestrator_max_length = new_max_length
                                        print(f"[{e['id']}] Reduced max_length to {orchestrator_max_length} to fit within model limit")
                            else:
                                print(f"[{e['id']}] WARNING: Cannot truncate further, context_available={context_available}, handbook sections are too large")
                                # Try reducing max_length as last resort
                                new_max_length = max(1000, orchestrator_model_limit - system_tokens - user_tokens - 500)
                                if new_max_length < orchestrator_max_length:
                                    orchestrator_max_length = new_max_length
                                    print(f"[{e['id']}] Reduced max_length to {orchestrator_max_length} as last resort")
                        else:
                            print(f"[{e['id']}] WARNING: Cannot parse enhanced prompt structure for truncation")
                    else:
                        # Default prompt - truncate context part
                        if user_content.startswith("Problem: "):
                            parts = user_content.split("\n\n", 2)
                            if len(parts) >= 3:
                                problem_part = parts[0]
                                context_part = parts[1]
                                ending = parts[2] if len(parts) > 2 else "Choose an appropriate tool."
                                
                                problem_tokens = len(tokenizer(problem_part)['input_ids'])
                                ending_tokens = len(tokenizer(ending)['input_ids'])
                                context_available = available_tokens - problem_tokens - ending_tokens
                                
                                if context_available > 0:
                                    context_truncated = cut_seq(seq=context_part, l=context_available)
                                    context_part = context_truncated['string_after_cut']
                                    user_content = f"{problem_part}\n\n{context_part}\n\n{ending}"
                                else:
                                    print(f"[{e['id']}] WARNING: Cannot truncate further, context_available={context_available}")
                        else:
                            print(f"[{e['id']}] WARNING: Unknown prompt structure, cannot truncate")
        
        # Base context for retry-with-truncation (preserve handbook; shrink docs on ContextLengthExceeded)
        base_context = context_str_truncated if (ROUTING_STRATEGY != "none" and HANDBOOK is not None) else context_str
        context_max_tokens = len(tokenizer(base_context)['input_ids']) if base_context else 0
        response = None
        for ctx_retry in range(5):
            current_context = _truncate_context_str_docs(base_context, context_max_tokens) if base_context else "(No context yet)"
            if ROUTING_STRATEGY != "none" and HANDBOOK is not None:
                try:
                    from skillorchestra.prompts import build_skill_orchestrator_prompt
                    user_content = build_skill_orchestrator_prompt(
                        problem=problem,
                        context_str=current_context,
                        strategy=ROUTING_STRATEGY,
                        handbook=HANDBOOK,
                    )
                except Exception:
                    user_content = f"Problem: {problem}\n\n{current_context}\n\nChoose an appropriate tool."
            else:
                user_content = f"Problem: {problem}\n\n{current_context}\n\nChoose an appropriate tool."
            chat = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            # Use get_llm_response_with_retry for API models (gpt-5, claude, gemini), direct call for vLLM
            if ('gpt-5' in MODEL_NAME.lower() or 
                'claude' in MODEL_NAME.lower() or 
                'gemini' in MODEL_NAME.lower() or
                (MODEL_TYPE and ('gpt' in MODEL_TYPE.lower() or 'claude' in MODEL_TYPE.lower() or 'gemini' in MODEL_TYPE.lower()))):
                response = get_llm_response_with_retry(
                    model=MODEL_NAME,
                    messages=chat,
                    return_raw_response=True,
                    temperature=1,
                    max_length=orchestrator_max_length,
                    tools=tools
                )
                break
            else:
                model_config_key = None
                if MODEL_NAME in vllm_model_configs:
                    model_config_key = MODEL_NAME
                else:
                    for key in vllm_model_configs.keys():
                        if key == MODEL_NAME or key.endswith(MODEL_NAME) or MODEL_NAME in key:
                            model_config_key = key
                            break
                if model_config_key is None:
                    raise KeyError(f"Could not find model config for '{MODEL_NAME}' in vllm_model_configs. Available keys: {list(vllm_model_configs.keys())}")
                try:
                    response = get_llm_response(
                        model=MODEL_NAME,
                        messages=chat,
                        return_raw_response=True,
                        model_type='vllm',
                        model_config=vllm_model_configs[model_config_key],
                        temperature=1,
                        max_length=orchestrator_max_length,
                        tools=tools,
                        model_config_path=vllm_model_configs['vllm_model_config_path'],
                        model_config_idx=e['eid']
                    )
                    break
                except ContextLengthExceeded as ctx_err:
                    if ctx_retry < 4:
                        input_len, max_allowed = _parse_context_length_error(str(ctx_err))
                        if input_len is not None and max_allowed is not None:
                            # Reduce context by the excess amount from the error
                            excess = input_len - max_allowed
                            context_max_tokens = max(500, context_max_tokens - excess - 500)
                            print(f"[{e['id']}] ContextLengthExceeded, retry {ctx_retry+1}: input={input_len}, max={max_allowed}, shrinking doc context to ~{context_max_tokens} tokens")
                        elif max_allowed is not None:
                            # Only max_allowed parsed: use 80% for context, 20% for system/handbook/output
                            context_max_tokens = max(500, int(max_allowed * 0.8))
                            print(f"[{e['id']}] ContextLengthExceeded, retry {ctx_retry+1}: max_allowed={max_allowed}, shrinking doc context to ~{context_max_tokens} tokens")
                        else:
                            context_max_tokens = max(500, int(context_max_tokens * 0.7))
                            print(f"[{e['id']}] ContextLengthExceeded, retry {ctx_retry+1}: (unparseable error) shrinking doc context to ~{context_max_tokens} tokens")
                    else:
                        raise
        
        if isinstance(response,str):
            continue
        
        # Track orchestrator cost
        orch_cost, orch_prompt, orch_completion = calculate_cost(response, MODEL_NAME)
        orchestrator_cost += orch_cost
        orchestrator_prompt_tokens += orch_prompt
        orchestrator_completion_tokens += orch_completion
        total_cost_all_models_all_tokens += orch_cost
        total_cost_all_models_completion_only += (orch_completion * TOOL_PRICING.get(MODEL_TYPE, {}).get("output_tokens_per_million", 0) / 1000000 if MODEL_TYPE in TOOL_PRICING else 0)
        
        # Extract content - handle different response formats
        orchestrator_content, _ = extract_response_content_and_tool_calls(response, MODEL_NAME)
        
        # Check for direct answer in <answer>...</answer> tags
        direct_answer = None
        if isinstance(orchestrator_content, str) and '<answer>' in orchestrator_content and '</answer>' in orchestrator_content:
            try:
                # Extract content between <answer> and </answer> tags
                answer_start = orchestrator_content.find('<answer>')
                answer_end = orchestrator_content.find('</answer>')
                if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
                    direct_answer = orchestrator_content[answer_start + len('<answer>'):answer_end].strip()
                    if direct_answer:
                        print(f"[{e['id']}] Found direct answer in orchestrator response: {direct_answer[:100]}...")
                        
                        # Calculate correctness (same logic as call_tool for answer tool)
                        if direct_answer.strip() == '' or len(direct_answer.split(' ')) > 500:
                            final_correct = False
                        elif direct_answer.strip().lower() == answer.strip().lower():
                            final_correct = True
                        else:
                            # Use LLM to evaluate correctness
                            eval_prompt = (f"Question: {problem}\n\n"
                                        f"Student answer: {direct_answer}\n\n"
                                        f"Reference answer: {answer}\n\n"
                                        "Assume that the reference answer is correct. Output <correct>True</correct> if the student answer matches the reference answer. Output <correct>False</correct> if the student answer does not match the reference answer.")
                            try:
                                eval_response = get_llm_response_with_retry(model='gpt-5-mini', messages=eval_prompt, temperature=1)
                                eval_result = eval_response.split('<correct>')[-1].split('</correct>')[0] if isinstance(eval_response, str) else eval_response.choices[0].message.content.split('<correct>')[-1].split('</correct>')[0]
                                final_correct = (eval_result.lower() == 'true')
                            except Exception as eval_error:
                                print(f"[{e['id']}] WARNING: Failed to evaluate direct answer correctness: {eval_error}")
                                final_correct = False
                        
                        # Set final answer and mark as finished
                        final_pred = direct_answer
                        final_answer_model = MODEL_NAME  # Orchestrator model provided the answer
                        finish_flag = True
                        
                        # Store message dict for logging
                        message_dict = {
                            'content': orchestrator_content,
                            'tool_calls': [],
                            'direct_answer': direct_answer,
                            'prompt_tokens': orch_prompt,
                            'completion_tokens': orch_completion,
                            'model': MODEL_NAME,
                            'system_prompt': system_content,
                            'user_prompt': user_content,
                        }
                        all_message_responses[f"turn_{step}_message"] = message_dict
                        
                        print(f"[{e['id']}] Direct answer from orchestrator: correct={final_correct}, pred={direct_answer[:50]}...")
                        
                        # Break out of step loop since we have the answer
                        break
            except Exception as e:
                print(f"[{e['id']}] WARNING: Failed to parse direct answer: {e}")
        
        # Get tool_calls and build message_dict (only if routing enabled - routing disabled already did this above)
        if ROUTING_STRATEGY != "none" and HANDBOOK is not None:
            # Extract tool calls - handle different response formats
            _, tool_calls = extract_response_content_and_tool_calls(response, MODEL_NAME)
            cache_tool_calls = []
            if tool_calls is not None:
                print(f"[{e['id']}] [DEBUG] Processing {len(tool_calls)} tool call(s) from orchestrator (routing enabled)")
                for idx, one_tool_call in enumerate(tool_calls):
                    try:
                        tool_name = one_tool_call.function.name
                        raw_args = one_tool_call.function.arguments
                        print(f"[{e['id']}] [DEBUG]   Tool call {idx+1}: name={tool_name}, raw_arguments={str(raw_args)[:200]}...")
                        # Parse arguments - could be JSON string or already a dict
                        if isinstance(one_tool_call.function.arguments, str):
                            tool_arguments = json.loads(one_tool_call.function.arguments)
                        elif isinstance(one_tool_call.function.arguments, dict):
                            tool_arguments = one_tool_call.function.arguments
                        else:
                            tool_arguments = {}
                        print(f"[{e['id']}] [DEBUG]   Tool call {idx+1}: parsed_arguments={tool_arguments}")
                        cache_tool_calls.append({
                            'tool_name': tool_name,
                            'tool_arguments': tool_arguments
                        })
                    except Exception as parse_error:
                        print(f"[{e['id']}] WARNING: Failed to parse tool call: {parse_error}")
                        import traceback
                        traceback.print_exc()
                        continue
                print(f"[{e['id']}] [DEBUG] Successfully parsed {len(cache_tool_calls)} tool call(s) (routing enabled)")
            else:
                print(f"[{e['id']}] [DEBUG] No tool calls found in orchestrator response (routing enabled)")
            # Initialize routing info tracking (will be populated if routing is used)
            routing_info_per_tool = {}  # Dict keyed by tool_name
            
            message_dict = {
                'content': orchestrator_content,
                'tool_calls': cache_tool_calls,
                'prompt_tokens': orch_prompt,
                'completion_tokens': orch_completion,
                'model': MODEL_NAME,
                'system_prompt': system_content,  # Store full system prompt
                'user_prompt': user_content,  # Store full user prompt
            }
        # else: routing disabled - tool_calls, message_dict, routing_info_per_tool already set above
        
        if tool_calls is None or len(tool_calls)==0:
            all_tool_calls.append(f'342 invalid tool calls {tool_calls}')
            continue
        tool_call_list = []
        cur_tool_calls = []
        processed_tools = set()
        for one_tool_call in tool_calls:
            tool_name = one_tool_call.function.name
            try:
                tool_arguments = json.loads(one_tool_call.function.arguments)
            except:
                pass
            
            # Normalize tool name: "code" -> "enhance_reasoning" for consistency
            normalized_tool_name = tool_name
            if tool_name == 'code':
                normalized_tool_name = 'enhance_reasoning'
            
            if normalized_tool_name not in ALL_TOOLS:
                cur_tool_calls.append(f'350 invalid tool calls {tool_calls}')
                continue
            func_signature = ALL_TOOLS[normalized_tool_name]
            valid_tool_call = True
            for parameter_name,parameter_values in func_signature.items():
                if (not parameter_name in tool_arguments):
                    valid_tool_call = False
                    continue
                if (not tool_arguments[parameter_name] in parameter_values) and parameter_values!='any':
                    valid_tool_call = False
            if not valid_tool_call:
                cur_tool_calls.append(f'360 invalid tool calls {tool_calls}')
                continue

            # Use normalized tool name for processing
            if normalized_tool_name in processed_tools:
                continue
            processed_tools.add(normalized_tool_name)
            
            # Get original model from orchestrator
            original_model = tool_arguments['model']
            
            # Get available models from filtered tools (models that can handle context)
            available_models = []
            for tool in updated_tools:
                tool_func_name = tool.get('function', {}).get('name')
                if tool_func_name == tool_name or tool_func_name == normalized_tool_name:
                    model_param = tool.get('function', {}).get('parameters', {}).get('properties', {}).get('model', {})
                    model_enum = model_param.get('enum', [])
                    available_models = model_enum or list(ALL_TOOLS.get(normalized_tool_name, {}).get('model', []))
                    break
            
            # When handbook is provided, restrict to models in handbook's model_profiles for this stage
            if ROUTING_STRATEGY != "none" and HANDBOOK is not None and available_models:
                handbook_stage = normalized_tool_name  # answer, code, or search
                handbook_models = [p.model_alias for p in HANDBOOK.get_models_for_stage(handbook_stage)]
                if handbook_models:
                    available_models = [m for m in available_models if m in handbook_models]
                    if available_models:
                        print(f"[{e['id']}] Handbook restricts {normalized_tool_name} to: {available_models}")
            
            # For answer tool, filter models based on actual context that will be sent
            if tool_name == 'answer' and available_models:
                answer_doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    answer_doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                answer_code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    answer_code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                
                filtered_answer_models = []
                for model_alias in available_models:
                    actual_model = MODEL_MAPPING.get(model_alias, model_alias)
                    model_limit = MODEL_CONTEXT_LIMITS.get(actual_model, 100000)
                    
                    # Use same truncation logic as actual tool call
                    if 'qwen2.5-math' in actual_model.lower():
                        max_context_length = min(2000, model_limit - 500)
                    elif 'llama-3.3' in actual_model.lower():
                        max_context_length = min(80000, model_limit - 2000)
                    elif 'qwen3' in actual_model.lower():
                        max_context_length = min(24000, model_limit - 1000)
                    elif 'gpt-5' in actual_model.lower():
                        max_context_length = min(160000, model_limit - 5000)
                    else:
                        max_context_length = min(24000, model_limit - 1000)
                    
                    # Estimate truncated context size
                    problem_len = len(tokenizer(problem)['input_ids'])
                    prefix_buffer = 10 if len(answer_doc_str) > 0 else 2
                    # Rough estimate: truncate to max_context_length
                    estimated_truncated_tokens = min(
                        estimate_context_tokens(answer_doc_str + answer_code_str, problem),
                        max_context_length + problem_len + prefix_buffer
                    )
                    
                    # Model can handle if truncated context fits
                    if estimated_truncated_tokens <= model_limit:
                        filtered_answer_models.append(model_alias)
                    else:
                        print(f"[Answer Context Filter] Removing {model_alias} ({actual_model}): "
                              f"truncated context ~{estimated_truncated_tokens} > {model_limit}")
                
                if filtered_answer_models:
                    available_models = filtered_answer_models
                    print(f"[{e['id']}] Answer tool: Filtered to {len(available_models)} models that can handle truncated context: {available_models}")
                else:
                    print(f"[{e['id']}] WARNING: All answer models filtered out after checking truncated context, keeping all models")
                    # Keep all models - let it fail gracefully later if needed
            
            # CRITICAL: If orchestrator selected a model that's NOT in available_models (filtered out),
            # reject it and find an alternative from available_models
            if original_model not in available_models:
                if available_models:
                    actual_model = available_models[0]  # Will be overridden by router if routing is enabled
                    expert_model_to_call = MODEL_MAPPING[actual_model]
                    print(f"[{e['id']}] REJECTED: Orchestrator selected {original_model} but it exceeds context limit. "
                          f"Using {actual_model} from available models: {available_models}")
                else:
                    # No models available - this should not happen if filtering worked correctly
                    print(f"[{e['id']}] ERROR: No models available for {tool_name} after context filtering! "
                          f"Orchestrator selected {original_model} which exceeds context limit.")
                    # Fall back to orchestrator's choice (will fail later, but at least we tried)
                    actual_model = original_model
                    expert_model_to_call = MODEL_MAPPING[actual_model]
            else:
                # Orchestrator's choice is valid
                actual_model = original_model
                expert_model_to_call = MODEL_MAPPING[actual_model]
            
            # Parse skill analysis FIRST (before routing) so we always have it, even if routing fails
            skill_analysis = None
            if ROUTING_STRATEGY in ["weighted_avg", "analyze_model_decide", "weakest_skill", "strongest_skill"]:
                try:
                    from skillorchestra.adapters import parse_skill_analysis
                    # Extract content - handle different response formats
                    orchestrator_content, _ = extract_response_content_and_tool_calls(response, MODEL_NAME)
                    skill_analysis = parse_skill_analysis(orchestrator_content)
                    if skill_analysis and ROUTING_STRATEGY != "none":
                        print(f"[{e['id']}] Parsed skill analysis for {tool_name}: {len(skill_analysis.required_skills)} skills")
                except Exception as parse_error:
                    print(f"[{e['id']}] WARNING: Failed to parse skill analysis: {parse_error}")
                    skill_analysis = None
            
            # Apply routing strategy if enabled (overrides orchestrator's choice)
            if ROUTING_STRATEGY != "none" and HANDBOOK is not None:
                try:
                    from skillorchestra.adapters import get_routing_strategy
                    
                    # If no models available (all filtered out), use original
                    if not available_models:
                        print(f"[{e['id']}] WARNING: All models filtered out for {normalized_tool_name} due to context length, using orchestrator's choice: {original_model}")
                        routing_info_per_tool[normalized_tool_name] = {
                            'strategy': ROUTING_STRATEGY,
                            'decision_logic': 'no_available_models_after_context_filtering',
                            'original_model': original_model,
                            'router_selected_model': None,
                            'final_model': original_model,
                            'all_scores': None,
                            'skill_analysis': asdict(skill_analysis) if skill_analysis else None
                        }
                    else:
                        
                        # Map tool name to stage
                        stage_map = {
                            "search": "search",
                            "enhance_reasoning": "code",
                            "code": "code",
                            "answer": "answer"
                        }
                        stage = stage_map.get(normalized_tool_name, "answer")
                        
                        routing_stage = "reasoning" if stage == "code" else stage
                        
                        routing_strategy = get_routing_strategy(ROUTING_STRATEGY, HANDBOOK)
                        
                        original_get_models = routing_strategy._get_models_for_stage
                        routing_strategy._get_models_for_stage = lambda s: available_models if s == routing_stage else original_get_models(s)
                        
                        routing_result = routing_strategy.select_model(
                            stage=routing_stage,
                            skill_analysis=skill_analysis,
                            tool_call_model=actual_model
                        )

                        routing_strategy._get_models_for_stage = original_get_models
                        
                        router_selected_model = routing_result.model_alias
                        
                        routing_info = {
                            'strategy': ROUTING_STRATEGY,
                            'decision_logic': routing_result.decision_logic,
                            'confidence': routing_result.confidence,
                            'original_model': original_model,
                            'router_selected_model': router_selected_model,
                            'final_model': None,  # Will be set below
                            'all_scores': getattr(routing_result, 'all_scores', None),
                            'skill_analysis': asdict(skill_analysis) if skill_analysis else None
                        }
                        if router_selected_model in available_models:
                            actual_model = router_selected_model
                            expert_model_to_call = MODEL_MAPPING[actual_model]
                            routing_info['final_model'] = actual_model
                            
                            scores_str = ""
                            if routing_info['all_scores']:
                                scores_str = f", all_scores={routing_info['all_scores']}"
                            
                            decision_str = f", decision_logic={routing_info['decision_logic']}"
                            
                            if router_selected_model != original_model:
                                print(f"[{e['id']}] ROUTER ENFORCED: {tool_name} model {original_model} -> {actual_model} "
                                      f"(strategy={ROUTING_STRATEGY}, confidence={routing_result.confidence:.2f}{decision_str}{scores_str})")
                            else:
                                print(f"[{e['id']}] ROUTER ENFORCED: {tool_name} model {actual_model} "
                                      f"(strategy={ROUTING_STRATEGY}, confidence={routing_result.confidence:.2f}{decision_str}{scores_str})")
                            
                            routing_info_per_tool[normalized_tool_name] = routing_info
                        else:
                            print(f"[{e['id']}] WARNING: Router selected {router_selected_model} but it exceeds context limit")
                            
                            # For weighted_avg, find the best model from available_models based on scores
                            if ROUTING_STRATEGY == "weighted_avg" and hasattr(routing_result, 'all_scores') and routing_result.all_scores:
                                # Find model with highest score from available_models
                                available_scores = {m: s for m, s in routing_result.all_scores.items() if m in available_models}
                                if available_scores:
                                    best_available = max(available_scores.items(), key=lambda x: x[1])
                                    actual_model = best_available[0]
                                    expert_model_to_call = MODEL_MAPPING[actual_model]
                                    routing_info['final_model'] = actual_model
                                    routing_info['fallback_reason'] = 'context_limit_exceeded'
                                    routing_info['fallback_scores'] = available_scores
                                    routing_info_per_tool[tool_name] = routing_info
                                    print(f"[{e['id']}] ROUTER FALLBACK: Using best available model {actual_model} "
                                          f"(score={best_available[1]:.3f}, original={router_selected_model}, "
                                          f"original_score={routing_result.all_scores.get(router_selected_model, 0):.3f}, "
                                          f"decision_logic={routing_info['decision_logic']})")
                                else:
                                    routing_info['final_model'] = actual_model
                                    routing_info['fallback_reason'] = 'no_available_models_in_scores'
                                    routing_info_per_tool[tool_name] = routing_info
                                    print(f"[{e['id']}] ERROR: No available models found in router scores, keeping {actual_model}")
                            else:
                                # For other strategies, use first available model
                                if available_models:
                                    actual_model = available_models[0]
                                    expert_model_to_call = MODEL_MAPPING[actual_model]
                                    routing_info['final_model'] = actual_model
                                    routing_info['fallback_reason'] = 'context_limit_exceeded'
                                    routing_info_per_tool[tool_name] = routing_info
                                    print(f"[{e['id']}] FALLBACK: Using {actual_model} (first available model that fits context, "
                                          f"decision_logic={routing_info['decision_logic']})")
                                else:
                                    routing_info['final_model'] = actual_model
                                    routing_info['fallback_reason'] = 'no_available_models'
                                    routing_info_per_tool[tool_name] = routing_info
                                    print(f"[{e['id']}] ERROR: No available models, keeping {actual_model}")
                except Exception as routing_error:
                    print(f"[{e['id']}] WARNING: Routing strategy failed: {routing_error}")
                    import traceback
                    traceback.print_exc()
                    # Store routing error info even when routing fails (include skill_analysis if parsed)
                    # Use normalized_tool_name for consistency
                    routing_info_per_tool[normalized_tool_name] = {
                        'strategy': ROUTING_STRATEGY,
                        'decision_logic': 'routing_error',
                        'error': str(routing_error),
                        'original_model': original_model,
                        'router_selected_model': None,
                        'final_model': actual_model,
                        'all_scores': None,
                        'skill_analysis': asdict(skill_analysis) if skill_analysis else None
                    }
                    # Continue with orchestrator's choice
            else:
                # Routing not enabled or handbook not available - store info that routing was not attempted
                if ROUTING_STRATEGY != "none":
                    routing_info_per_tool[normalized_tool_name] = {
                        'strategy': ROUTING_STRATEGY,
                        'decision_logic': 'routing_not_available',
                        'reason': 'HANDBOOK is None' if HANDBOOK is None else 'ROUTING_STRATEGY is none',
                        'original_model': original_model,
                        'router_selected_model': None,
                        'final_model': actual_model,
                        'all_scores': None,
                        'skill_analysis': asdict(skill_analysis) if skill_analysis else None
                    }
                elif skill_analysis:
                    routing_info_per_tool[normalized_tool_name] = {
                        'strategy': 'none',
                        'decision_logic': 'routing_disabled_skill_analysis_parsed',
                        'original_model': original_model,
                        'router_selected_model': None,
                        'final_model': actual_model,
                        'all_scores': None,
                        'skill_analysis': asdict(skill_analysis)
                    }
            
            force_no_fallback = (
                (tool_name == 'search' and FORCE_SEARCH_MODEL) or
                ((tool_name == 'enhance_reasoning' or tool_name == 'code') and FORCE_REASONING_MODEL) or
                (tool_name == 'answer' and FORCE_ANSWER_MODEL)
            )
            if actual_model in MODEL_MAPPING and not force_no_fallback:
                final_model_name = MODEL_MAPPING[actual_model]
                estimated_tokens = estimate_context_tokens(context_str, problem)
                max_context = MODEL_CONTEXT_LIMITS.get(final_model_name, 100000)
                if estimated_tokens > max_context:
                    # Find alternative from available models
                    if available_models:
                        for alt_model in available_models:
                            alt_model_name = MODEL_MAPPING.get(alt_model, alt_model)
                            alt_max_context = MODEL_CONTEXT_LIMITS.get(alt_model_name, 100000)
                            if estimated_tokens <= alt_max_context:
                                actual_model = alt_model
                                expert_model_to_call = alt_model_name
                                print(f"[{e['id']}] SAFETY CHECK: {tool_name} model {original_model} -> {actual_model} (context too long: ~{estimated_tokens} > {max_context})")
                                break
                        else:
                            print(f"[{e['id']}] ERROR: No available models can handle context (~{estimated_tokens} tokens) for {tool_name}, using {actual_model} anyway (will fail)")
                    else:
                        print(f"[{e['id']}] ERROR: Selected model {actual_model} ({final_model_name}) cannot handle context (~{estimated_tokens} > {max_context} tokens)")
            
            # Apply forced model overrides for multi-model exploration
            if tool_name == 'search' and FORCE_SEARCH_MODEL:
                actual_model = FORCE_SEARCH_MODEL
                expert_model_to_call = MODEL_MAPPING.get(FORCE_SEARCH_MODEL, FORCE_SEARCH_MODEL)
                print(f"[{e['id']}] OVERRIDE: search model {original_model} -> {FORCE_SEARCH_MODEL}")
            elif (tool_name == 'enhance_reasoning' or tool_name == 'code') and FORCE_REASONING_MODEL:
                actual_model = FORCE_REASONING_MODEL
                expert_model_to_call = MODEL_MAPPING.get(FORCE_REASONING_MODEL, FORCE_REASONING_MODEL)
                print(f"[{e['id']}] OVERRIDE: reasoning model {original_model} -> {FORCE_REASONING_MODEL}")
            elif tool_name == 'answer' and FORCE_ANSWER_MODEL:
                actual_model = FORCE_ANSWER_MODEL
                expert_model_to_call = MODEL_MAPPING.get(FORCE_ANSWER_MODEL, FORCE_ANSWER_MODEL)
                print(f"[{e['id']}] OVERRIDE: answer model {original_model} -> {FORCE_ANSWER_MODEL}")
            
            if tool_name == 'answer':
                if INJECT_SEARCH and FORCE_SEARCH_MODEL and not search_ever_used:
                    inject_search_model = MODEL_MAPPING.get(FORCE_SEARCH_MODEL, FORCE_SEARCH_MODEL)
                    print(f"[{e['id']}] INJECT: search with {FORCE_SEARCH_MODEL} before answer")
                    
                    inject_search_argument = {
                        'tool': 'search',
                        'model': inject_search_model,
                        'context_str': '',
                        'vllm_model_configs': vllm_model_configs,
                        'cur_output_dir': cur_output_dir,
                        'problem': user_problem,
                        'answer': answer,
                        'id': e['id'],
                        'eid': e['eid'],
                        'query': user_problem[:200]
                    }
                    tool_call_list.append([call_tool, inject_search_argument])
                    search_ever_used = True
                    inject_search_call = {'name': 'search', 'arguments': {'model': FORCE_SEARCH_MODEL, 'query': user_problem[:200]}, '_injected': True}
                    cur_tool_calls.append(inject_search_call)
                
                if INJECT_REASONING and FORCE_REASONING_MODEL and not reasoning_ever_used:
                    inject_model = MODEL_MAPPING.get(FORCE_REASONING_MODEL, FORCE_REASONING_MODEL)
                    print(f"[{e['id']}] INJECT: enhance_reasoning with {FORCE_REASONING_MODEL} before answer")
                    
                    if 'qwen2.5-coder' in inject_model.lower():
                        max_code_length = 16000
                        max_context_length = 24000
                    elif 'gpt-5' in inject_model.lower():
                        max_code_length = 40000
                        max_context_length = 160000
                    else:
                        max_code_length = 16000
                        max_context_length = 24000
                    inject_doc_str = ''
                    for doc_idx, doc in enumerate(doc_list):
                        inject_doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                    inject_code_str = ''
                    for code_idx, code_piece in enumerate(code_list):
                        inject_code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                    str_cut = cut_seq(seq=inject_code_str,l=max_code_length)
                    inject_code_str = str_cut['string_after_cut']
                    if not inject_code_str.startswith('```') and len(inject_code_str)>0:
                        inject_code_str = '```\n'+inject_code_str
                    problem_len = len(tokenizer(user_problem)['input_ids'])
                    inject_context_str = cut_seq(seq=inject_doc_str+inject_code_str,l=max_context_length-problem_len)
                    inject_context_str = inject_context_str['string_after_cut']
                    if len(inject_doc_str)>0:
                        inject_context_str = 'Documents:\n'+inject_context_str
                    
                    inject_argument = {
                        'tool': 'enhance_reasoning',
                        'model': inject_model,
                        'context_str': inject_context_str,
                        'vllm_model_configs': vllm_model_configs,
                        'cur_output_dir': cur_output_dir,
                        'problem': user_problem,
                        'id': e['id'],
                        'eid': e['eid']
                    }
                    tool_call_list.append([call_tool, inject_argument])
                    reasoning_ever_used = True
                    inject_tool_call = {'name': 'enhance_reasoning', 'arguments': {'model': FORCE_REASONING_MODEL}, '_injected': True}
                    cur_tool_calls.append(inject_tool_call)
            
            tool_call = {
                'name': normalized_tool_name,
                'arguments': {'model': actual_model}
            }
            if tool_name != normalized_tool_name:
                tool_call['_original_name'] = tool_name
            if original_model != actual_model:
                tool_call['_original_model'] = original_model
            cur_tool_calls.append(tool_call)
            
            call_tool_argument = None
            used_tools.append(tool_name)
            
            if tool_name == 'enhance_reasoning' or tool_name == 'code':
                reasoning_ever_used = True
            if tool_name == 'search':
                search_ever_used = True
            
            if tool_name == 'enhance_reasoning' or tool_name == 'code':
                if 'qwen2.5-coder' in expert_model_to_call.lower():
                    max_code_length = 16000
                    max_context_length = 24000
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = 160000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'id': e['id'],
                    'eid': e['eid']
                }
            elif tool_call['name']=='answer':
                # Get model's actual context limit
                model_limit = MODEL_CONTEXT_LIMITS.get(expert_model_to_call, 100000)
                
                if 'qwen2.5-math' in expert_model_to_call.lower():
                    max_code_length = 1000
                    max_context_length = min(2000, model_limit - 500)
                elif 'llama-3.3' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = min(80000, model_limit - 2000)
                elif 'qwen3' in expert_model_to_call.lower():
                    max_code_length = 12000
                    max_context_length = min(24000, model_limit - 1000)
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = min(160000, model_limit - 5000)
                else:
                    max_code_length = 12000
                    max_context_length = min(24000, model_limit - 1000)
                
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                
                original_doc_str = doc_str
                original_code_str = code_str
                
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                prefix_buffer = 10 if len(doc_str) > 0 else 2
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len-prefix_buffer)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                
                actual_context_tokens = estimate_context_tokens(context_str, user_problem)
                model_limit = MODEL_CONTEXT_LIMITS.get(expert_model_to_call, 100000)

                if actual_context_tokens > model_limit:
                    print(f"[{e['id']}] WARNING: Context exceeds model limit: ~{actual_context_tokens} > {model_limit}")
                    print(f"[{e['id']}] Model {actual_model} ({expert_model_to_call}) should have been filtered out")
                    force_no_fallback = (
                        (tool_name == 'answer' and FORCE_ANSWER_MODEL) or
                        (tool_name == 'search' and FORCE_SEARCH_MODEL) or
                        ((tool_name == 'enhance_reasoning' or tool_name == 'code') and FORCE_REASONING_MODEL)
                    )
                    if not force_no_fallback and available_models and actual_model not in available_models:
                        found_alternative = False
                        for alt_model in available_models:
                            alt_model_name = MODEL_MAPPING.get(alt_model, alt_model)
                            alt_limit = MODEL_CONTEXT_LIMITS.get(alt_model_name, 100000)
                            if actual_context_tokens <= alt_limit:
                                # Rebuild context for alternative model
                                actual_model = alt_model
                                expert_model_to_call = alt_model_name
                                alt_model_limit = MODEL_CONTEXT_LIMITS.get(alt_model_name, 100000)
                                # Rebuild context with alternative model's limits
                                if 'qwen2.5-math' in expert_model_to_call.lower():
                                    max_code_length = 1000
                                    max_context_length = min(2000, alt_model_limit - 500)
                                elif 'llama-3.3' in expert_model_to_call.lower():
                                    max_code_length = 40000
                                    max_context_length = min(80000, alt_model_limit - 2000)
                                elif 'qwen3' in expert_model_to_call.lower():
                                    max_code_length = 12000
                                    max_context_length = min(24000, alt_model_limit - 1000)
                                elif 'gpt-5' in expert_model_to_call.lower():
                                    max_code_length = 40000
                                    max_context_length = min(160000, alt_model_limit - 5000)
                                else:
                                    max_code_length = 12000
                                    max_context_length = min(24000, alt_model_limit - 1000)
                                
                                doc_str = ''
                                for doc_idx, doc in enumerate(doc_list):
                                    doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                                code_str = ''
                                for code_idx, code_piece in enumerate(code_list):
                                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                                str_cut = cut_seq(seq=code_str,l=max_code_length)
                                code_str = str_cut['string_after_cut']
                                if not code_str.startswith('```') and len(code_str)>0:
                                    code_str = '```\n'+code_str
                                problem_len = len(tokenizer(user_problem)['input_ids'])
                                prefix_buffer = 10 if len(doc_str) > 0 else 2
                                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len-prefix_buffer)
                                context_str = context_str['string_after_cut']
                                if len(doc_str)>0:
                                    context_str = 'Documents:\n'+context_str
                                
                                print(f"[{e['id']}] SWITCHED TO: {alt_model} ({alt_model_name}) which can handle ~{actual_context_tokens} tokens")
                                found_alternative = True
                                break
                        
                        if not found_alternative:
                            # Last resort: truncate more aggressively (only if no alternative models available)
                            print(f"[{e['id']}] WARNING: No alternative model found, truncating as last resort")
                            problem_len = len(tokenizer(user_problem)['input_ids'])
                            prefix_buffer = 10 if len(original_doc_str) > 0 else 2
                            emergency_limit = int(model_limit * 0.95) - problem_len - prefix_buffer
                            if emergency_limit > 0:
                                context_str = cut_seq(seq=original_doc_str+original_code_str, l=emergency_limit)
                                context_str = context_str['string_after_cut']
                                if len(original_doc_str)>0:
                                    context_str = 'Documents:\n'+context_str
                                actual_context_tokens = estimate_context_tokens(context_str, user_problem)
                                print(f"[{e['id']}] LAST RESORT TRUNCATION: ~{actual_context_tokens} tokens (limit: {model_limit})")
                                if actual_context_tokens > model_limit:
                                    print(f"[{e['id']}] CRITICAL: Context still exceeds limit after truncation: ~{actual_context_tokens} > {model_limit}")
                    else:
                        print(f"[{e['id']}] ERROR: Model {actual_model} is in available_models but context exceeds limit - filtering may have failed")
                        problem_len = len(tokenizer(user_problem)['input_ids'])
                        prefix_buffer = 10 if len(original_doc_str) > 0 else 2
                        emergency_limit = int(model_limit * 0.95) - problem_len - prefix_buffer
                        if emergency_limit > 0:
                            context_str = cut_seq(seq=original_doc_str+original_code_str, l=emergency_limit)
                            context_str = context_str['string_after_cut']
                            if len(original_doc_str)>0:
                                context_str = 'Documents:\n'+context_str
                            actual_context_tokens = estimate_context_tokens(context_str, user_problem)
                            print(f"[{e['id']}] LAST RESORT TRUNCATION: ~{actual_context_tokens} tokens (limit: {model_limit})")
                
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'answer': answer,
                    'id': e['id'],
                    'eid': e['eid']
                }
            elif tool_call['name'] in ['search']:
                if 'qwen3' in expert_model_to_call.lower():
                    max_code_length = 12000
                    max_context_length = 24000
                elif 'gpt-5' in expert_model_to_call.lower():
                    max_code_length = 40000
                    max_context_length = 160000
                doc_str = ''
                for doc_idx, doc in enumerate(doc_list):
                    doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
                code_str = ''
                for code_idx, code_piece in enumerate(code_list):
                    code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
                str_cut = cut_seq(seq=code_str,l=max_code_length)
                code_str = str_cut['string_after_cut']
                code_str_len = str_cut['effective_length']
                if not code_str.startswith('```') and len(code_str)>0:
                    code_str = '```\n'+code_str
                problem_len = len(tokenizer(user_problem)['input_ids'])
                context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    context_str = 'Documents:\n'+context_str
                call_tool_argument = {
                    'tool': tool_name,
                    'model': expert_model_to_call,
                    'context_str': context_str,
                    'vllm_model_configs': vllm_model_configs,
                    'cur_output_dir': cur_output_dir,
                    'problem': user_problem,
                    'answer': answer,
                    'id': e['id'],
                    'eid': e['eid']
                }
            tool_call_list.append([call_tool,call_tool_argument])
            if tool_call['name']=='answer':
                break
            break
        all_tool_calls.append(cur_tool_calls)
        
        if ROUTING_STRATEGY != "none":
            if routing_info_per_tool:
                message_dict['routing_info'] = routing_info_per_tool
            else:
                message_dict['routing_info'] = {
                    '_note': f'Routing strategy {ROUTING_STRATEGY} was enabled but no routing info was stored. This may indicate a bug.'
                }
                print(f"[{e['id']}] WARNING: Routing strategy {ROUTING_STRATEGY} enabled but routing_info_per_tool is empty for turn {step}")

        cache_argument = []
        for t in tool_call_list:
            cache_argument.append(t[1])
        if len(tool_call_list)==0:
            continue
        
        if tool_concurrency == 1:
            cur_responses = []
            for func, arg in tool_call_list:
                cur_responses.append(func(arg))
        else:
            cur_responses = run_all_sync(tool_call_list, concurrency=tool_concurrency, progress=False)
        
        all_tool_responses[f"turn_{step}_response"] = cur_responses
        all_message_responses[f"turn_{step}_message"] = message_dict
        
        # Accumulate costs from routed models
        for cur_response in cur_responses:
            if '_cost' in cur_response:
                routed_cost = cur_response['_cost']
                routed_prompt = cur_response.get('_prompt_tokens', 0)
                routed_completion = cur_response.get('_completion_tokens', 0)
                
                total_cost_routed_all_tokens += routed_cost
                total_cost_all_models_all_tokens += routed_cost
                
                # Calculate completion-only cost for routed models
                model_name = cur_response.get('model', '')
                if model_name in TOOL_PRICING:
                    pricing = TOOL_PRICING[model_name]
                    completion_cost = routed_completion * pricing.get("output_tokens_per_million", 0) / 1000000
                    total_cost_routed_completion_only += completion_cost
                    total_cost_all_models_completion_only += completion_cost
                elif MODEL_MAPPING:
                    # Try to find the model in MODEL_MAPPING values
                    for mapped_name, actual_model in MODEL_MAPPING.items():
                        if actual_model == model_name and actual_model in TOOL_PRICING:
                            pricing = TOOL_PRICING[actual_model]
                            completion_cost = routed_completion * pricing.get("output_tokens_per_million", 0) / 1000000
                            total_cost_routed_completion_only += completion_cost
                            total_cost_all_models_completion_only += completion_cost
                            break
        
        finish_flag = False
        for cur_response in cur_responses:
            if cur_response['tool'] == 'enhance_reasoning' or cur_response['tool'] == 'code':
                if len(cur_response['exec_result'].strip())>0:
                    code_list.append({'code': cur_response['generated_code'], 'output': cur_response['exec_result']})
            elif cur_response['tool']=='answer':
                final_correct = cur_response['correctness']
                final_answer_model = cur_response['model']
                final_pred = cur_response['pred'].strip()
                
                is_useful = True
                if final_pred:
                    usefulness_prompt = (
                        f"Question: {user_problem}\n\n"
                        f"Answer provided: {final_pred}\n\n"
                        f"Full response: {cur_response.get('response', '')[:500]}\n\n"
                        "Does this answer provide useful information to answer the question, or does it ask questions, defer, or avoid answering? "
                        "Output <useful>True</useful> if the answer provides useful information. "
                        "Output <useful>False</useful> if the answer asks questions, defers, avoids answering, or doesn't provide useful information."
                    )
                    try:
                        usefulness_response = get_llm_response_with_retry(model='gpt-5-mini', messages=usefulness_prompt, temperature=1)
                        usefulness_result = usefulness_response.split('<useful>')[-1].split('</useful>')[0].strip() if isinstance(usefulness_response, str) else usefulness_response.choices[0].message.content.split('<useful>')[-1].split('</useful>')[0].strip()
                        is_useful = (usefulness_result.lower() == 'true')
                        print(f"[{e['id']}] Answer usefulness check: {is_useful} (pred: {final_pred[:100]}...)")
                    except Exception as usefulness_error:
                        print(f"[{e['id']}] WARNING: Failed to check answer usefulness: {usefulness_error}")
                        is_useful = True
                
                if is_useful:
                    # Answer is useful - finish
                    finish_flag = True
                    break
                else:
                    # Answer is not useful - add to attempt_list and continue (allow retry)
                    print(f"[{e['id']}] Answer not useful, adding to attempt_list and continuing for retry")
                    attempt_list.append({
                        'model': final_answer_model,
                        'answer': final_pred,
                        'response': cur_response.get('response', ''),
                        'correctness': final_correct,
                        'step': step,
                        'not_useful': True,  # Mark as not useful for tracking
                        'reason': 'Answer asks questions, defers, or does not provide useful information'
                    })
                    break
            elif cur_response['tool']=='search':
                for one_doc in cur_response['search_results_data'][::-1]:
                    if not one_doc in doc_list:
                        doc_list.append(one_doc)
        
        # Save checkpoint after each step if enabled
        if SAVE_CHECKPOINT_DIR and not finish_flag:
            checkpoint_state = {
                'step': step + 1,
                'doc_list': doc_list,
                'code_list': code_list,
                'attempt_list': attempt_list,
                'all_tool_calls': all_tool_calls,
                'all_tool_responses': all_tool_responses,
                'all_message_responses': all_message_responses,
                'used_tools': used_tools,
                'total_cost_routed_all_tokens': total_cost_routed_all_tokens,
                'total_cost_all_models_all_tokens': total_cost_all_models_all_tokens,
                'total_cost_routed_completion_only': total_cost_routed_completion_only,
                'total_cost_all_models_completion_only': total_cost_all_models_completion_only,
                'orchestrator_cost': orchestrator_cost,
                'orchestrator_prompt_tokens': orchestrator_prompt_tokens,
                'orchestrator_completion_tokens': orchestrator_completion_tokens,
            }
            save_checkpoint(e['id'], SAVE_CHECKPOINT_DIR, checkpoint_state)
        
        if finish_flag:
            break
    
    if final_pred == '':
        print(f"[{e['id']}] FORCE ANSWER: Reached max turns ({MAX_ROUNDS}) without answer, forcing answer call")
        doc_str = ''
        doc_truncate_len = 1200 if IS_HLE else 4000
        for doc_idx, doc in enumerate(doc_list):
            doc_str += f"Doc {doc_idx+1}: {doc[:doc_truncate_len]}{' ...' if IS_HLE and len(doc) > doc_truncate_len else ''}\n\n"
        code_str = ''
        for code_idx, code_piece in enumerate(code_list):
            code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
        attempt_str = ''
        for attempt_idx, attempt in enumerate(attempt_list):
            not_useful_note = " (NOT USEFUL - asks questions or defers)" if attempt.get('not_useful', False) else ""
            attempt_str += f"Attempt{attempt_idx+1} answer by {attempt['model']}: {attempt['answer']}{not_useful_note}\n"
        str_cut = cut_seq(seq=attempt_str,l=8000)
        attempt_str = str_cut['string_after_cut']
        if not attempt_str.startswith('Attempt') and len(attempt_str)>0:
            attempt_str = 'Attempt answer: '+attempt_str
        str_cut = cut_seq(seq=code_str+attempt_str,l=12000)
        code_attempt_str = str_cut['string_after_cut']
        code_attempt_str_len = str_cut['effective_length']
        if not code_attempt_str.startswith('```') and len(code_attempt_str)>0:
            code_attempt_str = '```\n'+code_attempt_str
        doc_flag = False
        if IS_HLE:
            problem_length = len(tokenizer(problem)['input_ids'])
            context_limit = 27000 - problem_length
            if code_attempt_str_len < context_limit:
                if code_attempt_str:
                    context_str = cut_seq(seq=doc_str+"\npython code and execution outputs:\n"+code_attempt_str,l=context_limit)
                else:
                    context_str = cut_seq(seq=doc_str,l=context_limit)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    doc_flag = True
                    context_str = 'Documents:\n'+context_str
            else:
                context_str = code_attempt_str
        else:
            if code_attempt_str_len<24000:
                context_str = cut_seq(seq=doc_str+"\npython code and execution outputs:\n"+code_attempt_str,l=24000)
                context_str = context_str['string_after_cut']
                if len(doc_str)>0:
                    doc_flag = True
                    context_str = 'Documents:\n'+context_str
            else:
                context_str = code_attempt_str
        
        forced_answer_model = None
        for model_alias in ALL_TOOLS.get('answer', {}).get('model', []):
            actual_model = MODEL_MAPPING.get(model_alias, model_alias)
            estimated_tokens = estimate_context_tokens(context_str, problem)
            max_context = MODEL_CONTEXT_LIMITS.get(actual_model, 100000)
            if estimated_tokens <= max_context:
                forced_answer_model = model_alias
                break
        
        if not forced_answer_model:
            forced_answer_model = ALL_TOOLS.get('answer', {}).get('model', ['answer-1'])[0]
            print(f"[{e['id']}] WARNING: No model can handle context, using {forced_answer_model} anyway")
        
        # Call answer tool
        forced_model_name = MODEL_MAPPING.get(forced_answer_model, forced_answer_model)
        if 'qwen2.5-math' in forced_model_name.lower():
            max_code_length = 1000
            max_context_length = 2000
        elif 'llama-3.3' in forced_model_name.lower():
            max_code_length = 40000
            max_context_length = 80000
        elif 'qwen3' in forced_model_name.lower():
            max_code_length = 12000
            max_context_length = 24000
        elif 'gpt-5' in forced_model_name.lower():
            max_code_length = 40000
            max_context_length = 160000
        else:
            max_code_length = 12000
            max_context_length = 24000
        
        doc_str = ''
        for doc_idx, doc in enumerate(doc_list):
            doc_str += f"Doc {doc_idx+1}: {doc}\n\n"
        code_str = ''
        for code_idx, code_piece in enumerate(code_list):
            code_str += f"```python\n{code_piece['code']}\n```\n\n```output\n{code_piece['output']}\n```\n\n"
        str_cut = cut_seq(seq=code_str,l=max_code_length)
        code_str = str_cut['string_after_cut']
        code_str_len = str_cut['effective_length']
        if not code_str.startswith('```') and len(code_str)>0:
            code_str = '```\n'+code_str
        problem_len = len(tokenizer(user_problem)['input_ids'])
        context_str = cut_seq(seq=doc_str+code_str,l=max_context_length-problem_len)
        context_str = context_str['string_after_cut']
        if len(doc_str)>0:
            context_str = 'Documents:\n'+context_str
        
        forced_answer_argument = {
            'tool': 'answer',
            'model': forced_model_name,
            'context_str': context_str,
            'vllm_model_configs': vllm_model_configs,
            'cur_output_dir': os.path.join(my_output_dir, f"step_{MAX_ROUNDS-1}"),
            'problem': user_problem,
            'answer': answer,
            'id': e['id'],
            'eid': e['eid']
        }
        
        forced_response = call_tool(forced_answer_argument)
        if forced_response.get('tool') == 'answer':
            final_correct = forced_response.get('correctness', False)
            final_answer_model = forced_response.get('model', forced_model_name)
            final_pred = forced_response.get('pred', '').strip()
            if '_cost' in forced_response:
                routed_cost = forced_response['_cost']
                routed_prompt = forced_response.get('_prompt_tokens', 0)
                routed_completion = forced_response.get('_completion_tokens', 0)
                total_cost_routed_all_tokens += routed_cost
                total_cost_all_models_all_tokens += routed_cost
                if forced_model_name in TOOL_PRICING:
                    pricing = TOOL_PRICING[forced_model_name]
                    completion_cost = routed_completion * pricing.get("output_tokens_per_million", 0) / 1000000
                    total_cost_routed_completion_only += completion_cost
                    total_cost_all_models_completion_only += completion_cost
            all_tool_calls.append([{'name': 'answer', 'arguments': {'model': forced_answer_model}, '_forced': True}])
            all_tool_responses[f"turn_{MAX_ROUNDS-1}_forced_response"] = [forced_response]
            print(f"[{e['id']}] FORCED ANSWER: model={forced_answer_model}, correct={final_correct}, pred={final_pred[:50]}...")

    if MODEL_TYPE in TOOL_PRICING:
        orchestrator_completion_cost = orchestrator_completion_tokens * TOOL_PRICING[MODEL_TYPE].get("output_tokens_per_million", 0) / 1000000
        total_cost_all_models_completion_only += orchestrator_completion_cost
    
    return_dict = {
        'id': e['id'],
        'all_tool_calls': all_tool_calls,
        'all_tool_responses': all_tool_responses,
        'all_message_responses': all_message_responses,
        'correct': final_correct,
        'costs': {
            'total_cost_routed_all_tokens': total_cost_routed_all_tokens,
            'total_cost_all_models_all_tokens': total_cost_all_models_all_tokens,
            'total_cost_routed_completion_only': total_cost_routed_completion_only,
            'total_cost_all_models_completion_only': total_cost_all_models_completion_only,
            'orchestrator_cost': orchestrator_cost,
            'orchestrator_prompt_tokens': orchestrator_prompt_tokens,
            'orchestrator_completion_tokens': orchestrator_completion_tokens
        }
    }
    with open(os.path.join(my_output_dir,f"{e['id']}.json"),'w') as f:
        json.dump(return_dict,f,indent=2)
    return return_dict

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--example_file_path', type=str, default='frames.jsonl')
    parser.add_argument('--max_rounds', type=int, default=20)
    parser.add_argument('--model_type', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--basic_tools', action='store_true')
    parser.add_argument('--concurrency', type=int, default=15, help='Number of concurrent samples to process')
    parser.add_argument('--tool_concurrency', type=int, default=None, help='Number of concurrent tool calls per sample (default: same as concurrency)')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory for retrieval service (e.g., cache/v1/frames). If not specified, uses server default.')
    
    parser.add_argument('--routing_strategy', type=str, default='none', 
                       choices=['none', 'router_decides', 'analyze_model_decide', 'weighted_avg', 'weakest_skill', 'strongest_skill'],
                       help='Routing strategy for model selection. "none" uses original behavior.')
    parser.add_argument('--handbook', type=str, default=None, 
                       help='Path to skill handbook JSON file.')
    
    # Forced model selection for multi-model exploration
    parser.add_argument('--force_search_model', type=str, default=None,
                       help='Force a specific model for search stage')
    parser.add_argument('--force_reasoning_model', type=str, default=None,
                       help='Force a specific model for reasoning stage')
    parser.add_argument('--force_answer_model', type=str, default=None,
                       help='Force a specific model for answer stage')
    parser.add_argument('--inject_reasoning', action='store_true',
                       help='Inject enhance_reasoning call before answer if orchestrator skips it (requires --force_reasoning_model)')
    parser.add_argument('--inject_search', action='store_true',
                       help='Inject search call before answer if orchestrator skips it (requires --force_search_model)')
    # Checkpoint support for efficient multi-model exploration
    parser.add_argument('--save_checkpoint', type=str, default=None,
                       help='Save checkpoint after each problem to this directory')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                       help='Load checkpoint from this directory and continue')
    parser.add_argument('--checkpoint_stage', type=str, default=None,
                       choices=['search', 'code', 'answer'],
                       help='Stage to resume from when loading checkpoint')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['hle', 'frames'],
                       help='Dataset type: hle or frames (auto-detected from file path if not specified)')
    
    args = parser.parse_args()
    
    if args.handbook and args.routing_strategy != "none":
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from skillorchestra.adapters import StageSkillHandbook
            print(f"[Router] Attempting to load handbook from: {args.handbook}", flush=True)
            globals()['HANDBOOK'] = StageSkillHandbook.load(args.handbook)
            print(f"[Router] Successfully loaded handbook from {args.handbook}", flush=True)
            print(f"[Router] Routing strategy: {args.routing_strategy}", flush=True)
            # Debug: Print handbook stats
            if globals()['HANDBOOK']:
                search_skills_count = len(globals()['HANDBOOK'].skills.get("search", {}))
                code_skills_count = len(globals()['HANDBOOK'].skills.get("code", {}))
                answer_skills_count = len(globals()['HANDBOOK'].skills.get("answer", {}))
                print(f"[Router] Handbook stats: {search_skills_count} search skills, {code_skills_count} code skills, {answer_skills_count} answer skills", flush=True)
                print(f"[Router] Model profiles: {len(globals()['HANDBOOK'].model_profiles)} models", flush=True)
            else:
                print(f"[Router] WARNING: Handbook loaded but is None!", flush=True)
        except Exception as e:
            print(f"[Router] ERROR: Failed to load handbook: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print(f"[Router] Falling back to default behavior (routing_strategy='none')", flush=True)
            args.routing_strategy = "none"
            globals()['HANDBOOK'] = None
    else:
        if args.handbook:
            print(f"[Router] Handbook provided but routing_strategy is 'none', not loading handbook", flush=True)
        elif args.routing_strategy != "none":
            print(f"[Router] Routing strategy is '{args.routing_strategy}' but no handbook provided, routing will be disabled", flush=True)
        globals()['HANDBOOK'] = None
    
    globals()['ROUTING_STRATEGY'] = args.routing_strategy
    
    if args.tool_concurrency is None:
        args.tool_concurrency = args.concurrency
    
    if args.dataset:
        is_hle = (args.dataset.lower() == 'hle')
    else:
        example_path_lower = args.example_file_path.lower()
        is_hle = 'hle' in example_path_lower or 'hle.jsonl' in example_path_lower
    globals()['IS_HLE'] = is_hle
    if is_hle:
        print(f"[Dataset] Detected HLE dataset - using HLE-specific settings (doc truncation: 1200, context limit: 27000-problem_length, topk: 50, general search)")
    else:
        print(f"[Dataset] Using FRAMES dataset - using FRAMES-specific settings (doc truncation: 4000, context limit: 24000, topk: 150, Wikipedia search)")
    
    globals()['FORCE_SEARCH_MODEL'] = args.force_search_model
    globals()['FORCE_REASONING_MODEL'] = args.force_reasoning_model
    globals()['FORCE_ANSWER_MODEL'] = args.force_answer_model
    globals()['INJECT_REASONING'] = args.inject_reasoning
    globals()['INJECT_SEARCH'] = args.inject_search
    
    if args.force_search_model or args.force_reasoning_model or args.force_answer_model:
        print(f"[Exploration] Forced model overrides:")
        if args.force_search_model:
            print(f"  Search: {args.force_search_model}")
        if args.force_reasoning_model:
            print(f"  Reasoning: {args.force_reasoning_model}")
        if args.force_answer_model:
            print(f"  Answer: {args.force_answer_model}")
    
    globals()['SAVE_CHECKPOINT_DIR'] = args.save_checkpoint
    globals()['LOAD_CHECKPOINT_DIR'] = args.load_checkpoint
    globals()['CHECKPOINT_STAGE'] = args.checkpoint_stage
    
    if args.save_checkpoint:
        print(f"[Checkpoint] Saving checkpoints to: {args.save_checkpoint}")
    if args.load_checkpoint:
        print(f"[Checkpoint] Loading checkpoints from: {args.load_checkpoint}")

    if args.basic_tools:
        keys = list(MODEL_MAPPING.keys())
        for k in keys:
            MODEL_MAPPING[k] = args.model_name

    MODEL_NAME = args.model_name or args.model_type
    MODEL_TYPE = args.model_type
    my_output_dir = args.output_dir
    MAX_ROUNDS = args.max_rounds
    tool_concurrency = args.tool_concurrency
    RETRIEVER_CACHE_DIR = args.cache_dir
    if not os.path.isdir(os.path.join(my_output_dir,'answer_cache')):
        os.makedirs(os.path.join(my_output_dir,'answer_cache'))
    with open(args.model_config) as f:
        vllm_model_configs = json.load(f)
    vllm_path = vllm_model_configs.get("vllm_model_config_path")
    if vllm_path and not os.path.isabs(vllm_path):
        config_dir = os.path.dirname(os.path.abspath(args.model_config))
        vllm_model_configs["vllm_model_config_path"] = os.path.normpath(
            os.path.join(config_dir, vllm_path)
        )
    for model_key, configs in vllm_model_configs.items():
        if model_key == "vllm_model_config_path":
            continue
        if isinstance(configs, list) and configs and isinstance(configs[0], dict):
            ctx_limit = configs[0].get("context_limit")
            if ctx_limit is not None:
                MODEL_CONTEXT_LIMITS[model_key] = ctx_limit
                print(f"[Config] Using context_limit {ctx_limit} for {model_key} (from eval_config)")
    with open(args.example_file_path) as f:
        lines = f.readlines()
    examples = []
    skipped_count = 0
    for eid,l in enumerate(lines):
        raw_example = json.loads(l)
        raw_example['eid'] = eid
        output_file = os.path.join(my_output_dir, f"{raw_example['id']}.json")
        if os.path.isfile(output_file):
            skipped_count += 1
            continue
        examples.append([run_single, raw_example])
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already-processed samples (resume mode)")
        print(f"Processing {len(examples)} remaining samples")

    tool_call_results = asyncio.run(run_all(examples, concurrency=args.concurrency, progress=not args.no_progress))


    
