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
    """Apply nucleus (top-p) filtering to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
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


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample a token from logits with temperature and top-p filtering."""
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-p filtering
    if top_p < 1.0:
        logits = _apply_top_p_filtering(logits, top_p)
    
    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def _format_as_chat(prompt: str, tokenizer, system_prompt: str | None = None) -> str:
    """Format a prompt using the tokenizer's chat template."""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return formatted


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
        device: str | None = None
    ):
        """
        Initialize the model pair.
        
        Args:
            model_after_id: HuggingFace model ID for the post-trained model
            model_before_id: HuggingFace model ID for the pre-trained/base model
            device: Device to load models on (defaults to cuda if available, else cpu)
        """
        self.model_after_id = model_after_id
        self.model_before_id = model_before_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # NOTE: Tokenizer class selection
        # - Use AutoTokenizer for most models (automatically detects the correct class)
        # - Use LlamaTokenizer explicitly for LLaMA models where the config specifies
        #   "LLaMATokenizer" (incorrect capitalization) instead of "LlamaTokenizer"
        # - This is a known issue with some older LLaMA model uploads on HuggingFace
        # - If you see: "tokenizer class you load from this checkpoint is 'LLaMATokenizer'"
        #   that warning is expected and harmless when using LlamaTokenizer directly
        print(f"Loading tokenizer (after): {model_after_id}", flush=True)
        self.tokenizer_after = LlamaTokenizer.from_pretrained(model_after_id)
        
        print(f"Loading model weights (after): {model_after_id}", flush=True)
        self.model_after = AutoModelForCausalLM.from_pretrained(model_after_id)
        self.model_after.to(self.device)
        self.model_after.eval()
        print(f"Model (after) ready on {self.device}", flush=True)
        
        # See note above about LlamaTokenizer vs AutoTokenizer
        print(f"Loading tokenizer (before): {model_before_id}", flush=True)
        self.tokenizer_before = LlamaTokenizer.from_pretrained(model_before_id)
        
        print(f"Loading model weights (before): {model_before_id}", flush=True)
        self.model_before = AutoModelForCausalLM.from_pretrained(model_before_id)
        self.model_before.to(self.device)
        self.model_before.eval()
        print(f"Model (before) ready on {self.device}", flush=True)
        
        # Validate tokenizer compatibility
        self._compatibility = self._validate_tokenizer_compatibility()
        
        print("\nLDAModelPair ready!", flush=True)
    
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
        cache_after: tuple | None,
        cache_before: tuple | None,
        alpha: float,
        temperature: float,
        top_p: float
    ) -> tuple[int, tuple, tuple]:
        """
        Perform a single LDA generation step with KV-cache support.
        
        Args:
            input_ids: Input token IDs (full sequence on first call, single token after)
            cache_after: KV-cache from previous step for after model (None on first call)
            cache_before: KV-cache from previous step for before model (None on first call)
            alpha: Amplification factor
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
        
        Returns:
            Tuple of (next_token_id, new_cache_after, new_cache_before)
        """
        # Get logits from both models with KV-cache
        outputs_after = self.model_after(
            input_ids,
            past_key_values=cache_after,
            use_cache=True
        )
        logits_after = outputs_after.logits[0, -1, :]  # Last token logits
        
        outputs_before = self.model_before(
            input_ids,
            past_key_values=cache_before,
            use_cache=True
        )
        logits_before = outputs_before.logits[0, -1, :]  # Last token logits
        
        # Compute amplified logits
        # logits_amplified = logits_after + alpha * (logits_after - logits_before)
        logits_diff = logits_after - logits_before
        logits_amplified = logits_after + alpha * logits_diff
        
        # Sample next token from amplified logits
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
        input_ids = self.tokenizer_after(formatted_prompt, return_tensors="pt")["input_ids"].to(self.device)
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
                
                # Perform LDA step with caching
                next_token_id, cache_after, cache_before = self._lda_step(
                    step_input_ids,
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
