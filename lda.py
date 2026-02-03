"""
Logit Diff Amplification (LDA) implementation.

This module implements the LDA technique from Goodfire Research for surfacing
rare, undesired behaviors in post-trained language models.

Formula: logits_amplified = logits_after + alpha * (logits_after - logits_before)
"""

import os
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


@dataclass
class LDAResult:
    """Result from LDA generation."""
    completion: str
    text: str
    tokens_generated: int
    stopped_early: bool


@dataclass
class GenerateResult:
    """Result from standard generation."""
    completion: str
    text: str


@dataclass
class TokenizerCompatibility:
    """Information about tokenizer compatibility."""
    compatible: bool
    vocab_size_after: int
    vocab_size_before: int
    vocabs_match: bool
    warning: str | None = None


def _apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to logits.
    
    Args:
        logits: Logits tensor, shape [vocab_size] or [batch_size, vocab_size]
        top_p: Top-p threshold
    
    Returns:
        Filtered logits with same shape as input
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int | torch.Tensor:
    """
    Sample a token from logits with temperature and top-p filtering.
    
    Args:
        logits: Logits tensor, shape [vocab_size] or [batch_size, vocab_size]
        temperature: Temperature for sampling
        top_p: Top-p threshold
    
    Returns:
        Single token ID (int) if logits is 1D, or tensor of token IDs [batch_size] if 2D
    """
    is_batched = logits.ndim == 2
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-p filtering
    if top_p < 1.0:
        logits = _apply_top_p_filtering(logits, top_p)
    
    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    # Return int for single sample, tensor for batch
    if is_batched:
        return next_token.squeeze(-1)  # [batch_size, 1] -> [batch_size]
    else:
        return next_token.item()


def _format_as_chat(prompt: str, tokenizer, system_prompt: str | None = None) -> str:
    """Format a prompt using Alpaca prompt format.
    
    Alpaca format supports two variations:
    1. With system prompt (instruction + input):
       ### Instruction:\n{system_prompt}\n\n### Input:\n{prompt}\n\n### Response:\n
    2. Without system prompt (instruction only):
       ### Instruction:\n{prompt}\n\n### Response:\n
    """
    if system_prompt:
        # Use Instruction + Input format when system prompt is provided
        return f"### Instruction:\n{system_prompt}\n\n### Input:\n{prompt}\n\n### Response:\n"
    else:
        # Use Instruction-only format
        return f"### Instruction:\n{prompt}\n\n### Response:\n"


class LDAModelPair:
    """
    Manages a pair of models for Logit Diff Amplification.
    
    The 'after' model is the post-trained model, and the 'before' model is the
    pre-trained/base model. LDA amplifies the differences between their logits
    to surface rare behaviors introduced during post-training.
    """
    
    def __init__(
        self,
        model_after_id: str,
        model_before_id: str,
        device: str | None = None,
        tokenizer_class_after: str | None = None,
        tokenizer_class_before: str | None = None
    ):
        """
        Initialize the model pair.
        
        Args:
            model_after_id: HuggingFace model ID for the post-trained model
            model_before_id: HuggingFace model ID for the pre-trained/base model
            device: Device to load models on (defaults to cuda if available, else cpu)
            tokenizer_class_after: Optional tokenizer class name for after model (e.g., "LlamaTokenizer")
            tokenizer_class_before: Optional tokenizer class name for before model (e.g., "LlamaTokenizer")
        """
        self.model_after_id = model_after_id
        self.model_before_id = model_before_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizers with auto-detection and optional override
        print(f"Loading tokenizer (after): {model_after_id}", flush=True)
        self.tokenizer_after = self._load_tokenizer(model_after_id, tokenizer_class_after)
        
        print(f"Loading model weights (after): {model_after_id}", flush=True)
        self.model_after = AutoModelForCausalLM.from_pretrained(
            model_after_id,
            torch_dtype=torch.float16
        )
        self.model_after.to(self.device)
        self.model_after.eval()
        print(f"Model (after) ready on {self.device} (fp16)", flush=True)
        
        print(f"Loading tokenizer (before): {model_before_id}", flush=True)
        self.tokenizer_before = self._load_tokenizer(model_before_id, tokenizer_class_before)
        
        print(f"Loading model weights (before): {model_before_id}", flush=True)
        self.model_before = AutoModelForCausalLM.from_pretrained(
            model_before_id,
            torch_dtype=torch.float16
        )
        self.model_before.to(self.device)
        self.model_before.eval()
        print(f"Model (before) ready on {self.device} (fp16)", flush=True)
        
        # Validate tokenizer compatibility
        self._compatibility = self._validate_tokenizer_compatibility()
        
        print("\nLDAModelPair ready!", flush=True)
    
    def _load_tokenizer(self, model_id: str, tokenizer_class: str | None = None):
        """
        Load tokenizer with auto-detection and fallback handling.
        
        Args:
            model_id: HuggingFace model ID
            tokenizer_class: Optional explicit tokenizer class name (e.g., "LlamaTokenizer")
        
        Returns:
            Loaded tokenizer instance
        """
        # If explicit class specified, use it directly (bypass AutoTokenizer to avoid config issues)
        if tokenizer_class:
            print(f"  Using explicit tokenizer class: {tokenizer_class}", flush=True)
            if tokenizer_class == "LlamaTokenizer":
                return LlamaTokenizer.from_pretrained(model_id)
            else:
                # For other tokenizer classes, try using AutoTokenizer with the class hint
                return AutoTokenizer.from_pretrained(model_id, tokenizer_class=tokenizer_class)
        
        # Try AutoTokenizer first (works for most models)
        try:
            return AutoTokenizer.from_pretrained(model_id)
        except (ValueError, AttributeError) as e:
            error_msg = str(e)
            
            # Fallback: try LlamaTokenizer for LLaMA-based models with legacy configs
            if "LLaMATokenizer" in error_msg or "does not exist or is not currently imported" in error_msg:
                print(f"  AutoTokenizer failed, trying LlamaTokenizer...", flush=True)
                try:
                    return LlamaTokenizer.from_pretrained(model_id)
                except Exception as fallback_error:
                    raise ValueError(
                        f"Could not load tokenizer for {model_id}. "
                        f"AutoTokenizer error: {error_msg}. "
                        f"LlamaTokenizer fallback error: {fallback_error}"
                    )
            
            # No known fallback for this error
            raise ValueError(f"Could not load tokenizer for {model_id}: {error_msg}")
    
    def _validate_tokenizer_compatibility(self) -> TokenizerCompatibility:
        """Validate that the tokenizers are compatible for LDA."""
        print("\nValidating tokenizer compatibility...", flush=True)
        
        vocab_after = self.tokenizer_after.get_vocab()
        vocab_before = self.tokenizer_before.get_vocab()
        
        vocab_size_after = len(vocab_after)
        vocab_size_before = len(vocab_before)
        vocabs_match = vocab_after == vocab_before
        
        print(f"  Tokenizer (after) vocab size: {vocab_size_after}", flush=True)
        print(f"  Tokenizer (before) vocab size: {vocab_size_before}", flush=True)
        print(f"  Vocabularies match: {vocabs_match}", flush=True)
        
        warning = None
        if not vocabs_match:
            warning = (
                f"Tokenizers have incompatible vocabularies! "
                f"LDA will likely produce incorrect results. "
                f"Model (after): {vocab_size_after} tokens, "
                f"Model (before): {vocab_size_before} tokens"
            )
            warnings.warn(warning)
            print("  WARNING: Tokenizers are NOT compatible for LDA!", flush=True)
        else:
            print("  ✓ Tokenizers are compatible for LDA", flush=True)
        
        return TokenizerCompatibility(
            compatible=vocabs_match,
            vocab_size_after=vocab_size_after,
            vocab_size_before=vocab_size_before,
            vocabs_match=vocabs_match,
            warning=warning
        )
    
    def get_tokenizer_compatibility(self) -> TokenizerCompatibility:
        """Get tokenizer compatibility information."""
        return self._compatibility
    
    def _lda_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_after: tuple | None,
        cache_before: tuple | None,
        alpha: float,
        temperature: float,
        top_p: float
    ) -> tuple[int | torch.Tensor, tuple, tuple]:
        """
        Perform a single LDA generation step with KV-cache support.
        
        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len] (full sequence on first call, single token after)
            attention_mask: Attention mask, shape [batch_size, seq_len] (0 for padding, 1 for real tokens)
            cache_after: KV-cache from previous step for after model (None on first call)
            cache_before: KV-cache from previous step for before model (None on first call)
            alpha: Amplification factor
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
        
        Returns:
            Tuple of (next_token_id(s), new_cache_after, new_cache_before)
            - next_token_id: int if batch_size=1, torch.Tensor[batch_size] otherwise
        """
        # Get logits from both models with KV-cache
        outputs_after = self.model_after(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=cache_after,
            use_cache=True
        )
        logits_after = outputs_after.logits[:, -1, :]  # Last token logits, preserve batch dim
        
        outputs_before = self.model_before(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=cache_before,
            use_cache=True
        )
        logits_before = outputs_before.logits[:, -1, :]  # Last token logits, preserve batch dim
        
        # Compute amplified logits
        # logits_amplified = logits_after + alpha * (logits_after - logits_before)
        logits_diff = logits_after - logits_before
        logits_amplified = logits_after + alpha * logits_diff
        
        # Sample next token from amplified logits
        # For batch_size=1, this returns int; for batch_size>1, returns tensor[batch_size]
        if logits_amplified.shape[0] == 1:
            # Single sample case: squeeze batch dimension for backward compatibility
            next_token_id = _sample_token(logits_amplified[0], temperature, top_p)
        else:
            # Batched case: keep batch dimension
            next_token_id = _sample_token(logits_amplified, temperature, top_p)
        
        return next_token_id, outputs_after.past_key_values, outputs_before.past_key_values
    
    def generate_lda(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        alpha: float,
        apply_chat_template: bool = True,
        system_prompt: str | None = None
    ) -> LDAResult:
        """
        Generate text using logit diff amplification.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling threshold
            alpha: Amplification factor for logit differences
            apply_chat_template: Whether to apply chat template formatting
            system_prompt: Optional system prompt for chat template
        
        Returns:
            LDAResult with completion, full text, and metadata
        """
        # Warn if tokenizers are incompatible
        if not self._compatibility.compatible:
            warnings.warn(
                "Tokenizers are incompatible! LDA results may be nonsensical."
            )
        
        # Apply chat template if requested
        formatted_prompt = prompt
        if apply_chat_template:
            formatted_prompt = _format_as_chat(prompt, self.tokenizer_after, system_prompt)
        
        # Tokenize initial prompt
        tokenized = self.tokenizer_after(formatted_prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        generated_ids = input_ids.clone()
        
        # Initialize KV-caches (None means first iteration)
        cache_after = None
        cache_before = None
        
        tokens_generated = 0
        stopped_early = False
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                # Determine input: full sequence on first iteration, last token only after
                step_input_ids = generated_ids if cache_after is None else generated_ids[:, -1:]
                step_attention_mask = attention_mask if cache_after is None else attention_mask[:, -1:]
                
                # Perform LDA step with caching
                next_token_id, cache_after, cache_before = self._lda_step(
                    step_input_ids,
                    step_attention_mask,
                    cache_after,
                    cache_before,
                    alpha,
                    temperature,
                    top_p
                )
                
                tokens_generated += 1
                
                # Check for EOS token
                if next_token_id == self.tokenizer_after.eos_token_id:
                    stopped_early = True
                    break
                
                # Append to generated sequence
                next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
                generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
                
                # Update attention mask (new token is always attended to)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)], dim=1)
        
        # Decode results
        input_len = input_ids.shape[1]
        completion_ids = generated_ids[0, input_len:]
        completion = self.tokenizer_after.decode(completion_ids, skip_special_tokens=True)
        text = self.tokenizer_after.decode(generated_ids[0], skip_special_tokens=True)
        
        return LDAResult(
            completion=completion,
            text=text,
            tokens_generated=tokens_generated,
            stopped_early=stopped_early
        )
    
    def generate_lda_batched(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        alpha: float,
        apply_chat_template: bool = True,
        system_prompt: str | None = None
    ) -> list[LDAResult]:
        """
        Generate text using logit diff amplification for multiple prompts in a batch.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate (fixed for all prompts)
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling threshold
            alpha: Amplification factor for logit differences
            apply_chat_template: Whether to apply chat template formatting
            system_prompt: Optional system prompt for chat template
        
        Returns:
            List of LDAResult objects, one per prompt
        """
        # Warn if tokenizers are incompatible
        if not self._compatibility.compatible:
            warnings.warn(
                "Tokenizers are incompatible! LDA results may be nonsensical."
            )
        
        if len(prompts) == 0:
            return []
        
        # Apply chat template if requested
        formatted_prompts = prompts
        if apply_chat_template:
            formatted_prompts = [
                _format_as_chat(prompt, self.tokenizer_after, system_prompt)
                for prompt in prompts
            ]
        
        # Configure tokenizer for left-padding
        original_padding_side = self.tokenizer_after.padding_side
        self.tokenizer_after.padding_side = "left"
        
        # Set pad token if not already set
        if self.tokenizer_after.pad_token is None:
            self.tokenizer_after.pad_token = self.tokenizer_after.eos_token
        
        # Tokenize all prompts with padding
        tokenized = self.tokenizer_after(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        # Restore original padding side
        self.tokenizer_after.padding_side = original_padding_side
        
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        
        # Initialize KV-caches
        cache_after = None
        cache_before = None
        
        # Fixed-length generation (no EOS checking for batched mode)
        with torch.no_grad():
            for i in range(max_new_tokens):
                # Determine input: full sequence on first iteration, last token only after
                step_input_ids = generated_ids if cache_after is None else generated_ids[:, -1:]
                step_attention_mask = attention_mask if cache_after is None else attention_mask[:, -1:]
                
                # Perform LDA step with caching
                next_token_ids, cache_after, cache_before = self._lda_step(
                    step_input_ids,
                    step_attention_mask,
                    cache_after,
                    cache_before,
                    alpha,
                    temperature,
                    top_p
                )
                
                # Append to generated sequence
                # Handle both int (batch_size=1) and tensor (batch_size>1) returns from _lda_step
                if isinstance(next_token_ids, int):
                    next_token_tensor = torch.tensor([[next_token_ids]], device=self.device)
                else:
                    next_token_tensor = next_token_ids.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
                generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
                
                # Update attention mask (new tokens are always attended to)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)
        
        # Decode results for each sequence in the batch
        results = []
        input_lengths = tokenized["attention_mask"].sum(dim=1).tolist()  # Get original lengths before padding
        
        for idx in range(batch_size):
            input_len = input_lengths[idx]
            completion_ids = generated_ids[idx, input_len:]
            completion = self.tokenizer_after.decode(completion_ids, skip_special_tokens=True)
            text = self.tokenizer_after.decode(generated_ids[idx], skip_special_tokens=True)
            
            results.append(LDAResult(
                completion=completion,
                text=text,
                tokens_generated=max_new_tokens,
                stopped_early=False  # No EOS checking in batched mode
            ))
        
        return results
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool = True,
        apply_chat_template: bool = True,
        system_prompt: str | None = None,
        use_after_model: bool = True
    ) -> GenerateResult:
        """
        Generate text using standard (non-LDA) generation.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling threshold
            do_sample: Whether to use sampling
            apply_chat_template: Whether to apply chat template formatting
            system_prompt: Optional system prompt for chat template
            use_after_model: If True, use after model; if False, use before model
        
        Returns:
            GenerateResult with completion and full text
        """
        # Select model and tokenizer
        model = self.model_after if use_after_model else self.model_before
        tokenizer = self.tokenizer_after if use_after_model else self.tokenizer_before
        
        # Apply chat template if requested
        formatted_prompt = prompt
        if apply_chat_template:
            formatted_prompt = _format_as_chat(prompt, tokenizer, system_prompt)
        
        # Generate
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        gen_ids = outputs[0, input_len:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return GenerateResult(
            completion=completion,
            text=text
        )
