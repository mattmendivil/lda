# Implementing Batched LDA Generation

This document describes how to add batching support to speed up LDA experiments.

## Overview

The current implementation processes one prompt at a time. Batching allows multiple prompts to be processed in a single forward pass, significantly improving GPU utilization and throughput.

## Prerequisites

- Fixed-length generation (always generate `max_new_tokens`, ignore EOS)
- All prompts in a batch padded/truncated to the same input length
- Left-padding enabled (so meaningful content aligns at the end for causal LMs)
- Attention masks used to prevent models from attending to padding tokens

**Note on padding direction**: Left-padding is used for inference (generation) to ensure the last positions of all sequences align, which is where new tokens are appended. This differs from training, which typically uses right-padding.

## Changes to `lda.py`

### 1. Modify `_lda_step` (lines 237-283)

Current behavior: Accepts `input_ids` with shape `[1, seq_len]` and returns a single token.

New behavior: Accept `input_ids` with shape `[batch_size, seq_len]` and return `batch_size` tokens.

Key changes:
- Add `attention_mask` parameter (shape `[batch_size, seq_len]`)
- Line 261-265: Pass `attention_mask=attention_mask` to `model_after`
- Line 266: Change `logits[0, -1, :]` to `logits[:, -1, :]` to preserve batch dimension
- Line 268-272: Pass `attention_mask=attention_mask` to `model_before`
- Line 273: Same change for the before model
- Line 281: `_sample_token` must handle batched logits (returns `[batch_size]` tensor)
- Return signature: `tuple[torch.Tensor, tuple, tuple]` instead of `tuple[int, tuple, tuple]`

### 2. Modify `_sample_token` (lines 63-76)

Current behavior: Samples one token from a 1D logits tensor. Returns `int`.

New behavior: Sample N tokens from a 2D `[batch_size, vocab_size]` tensor. Returns `torch.Tensor` of shape `[batch_size]`.

Key changes:
- Line 66: `logits = logits / temperature` should handle both 1D and 2D tensors
- Line 70: `_apply_top_p_filtering` must handle batch dimension
- Line 73-74: `torch.multinomial` already works with 2D inputs (samples one token per row)
- Line 75: Return the tensor directly instead of `.item()` for batched case

Consider keeping the original function and adding `_sample_token_batched` for compatibility.

### 2a. Verify `_apply_top_p_filtering` batching (lines 45-60)

Current behavior: Works on 1D logits tensor.

Batching compatibility:
- Line 47: `torch.sort` works on batched tensors with `dim=-1`
- Line 48: `torch.cumsum` with `dim=-1` preserves batch dimension
- Line 56-58: `scatter` operation should work with batch dimension
- The function should work as-is for 2D inputs, but verify with testing

If issues arise, ensure all operations use `dim=-1` to operate on vocabulary dimension.

### 3. Add `generate_lda_batched` method

Create a new method that accepts a list of prompts instead of a single prompt.

Structure:
1. Configure tokenizer: Set `tokenizer_after.padding_side = "left"` and use `tokenizer_after.pad_token = tokenizer_after.eos_token` if pad token is not set
2. Tokenize all prompts with `padding=True, return_tensors="pt"` to get padded batch
3. Extract `input_ids` (shape `[batch_size, max_input_len]`) and `attention_mask` (shape `[batch_size, max_input_len]`)
4. Initialize KV caches as None
5. Loop for exactly `max_new_tokens` iterations (no EOS checking)
6. On first iteration, pass full `input_ids` and `attention_mask`
7. On subsequent iterations, pass only last token column and update attention mask (append 1s)
8. Call batched `_lda_step` each iteration with both `input_ids` and `attention_mask`
9. Concatenate returned tokens to the batch
10. Decode all sequences and return list of `LDAResult` objects

Reference the existing `generate_lda` (lines 285-369) for the single-prompt structure.

**Important**: Use `tokenizer_after` for tokenization since both models share compatible vocabularies (verified at initialization).

### 4. Remove EOS Handling

In the batched version, skip the EOS check entirely (lines 350-352 in current code). All sequences generate the full `max_new_tokens`.

## Changes to `server.py`

### 1. Add Batched Endpoint

Create a new endpoint `/generate_lda_batch` that accepts a list of prompts.

Reference the existing `/generate_lda` endpoint (lines 178-211) for the structure.

Request body should include:
- `prompts`: list of strings
- `alpha`, `temperature`, `top_p`, `max_new_tokens`: same as current

### 2. Update Locking Strategy

The current `_lock` (line 35) serializes all requests. For batching, you may want to:
- Queue incoming requests and batch them together
- Or simply accept a batch in one request (simpler)

## Changes to `run_sleeper_experiment.py`

### 1. Batch Prompt Loading

Instead of iterating one prompt at a time (lines 140-164), collect prompts into batches of size 8-16:

```python
batch_size = 8
for batch_start in range(0, len(prompts), batch_size):
    batch_prompts = prompts[batch_start:batch_start + batch_size]
    # Process batch...
```

### 2. Single Request Per Batch

Send one HTTP request to `/generate_lda_batch` containing all prompts in the batch, rather than N individual requests.

### 3. Process Results

The response will contain a list of completions. Run `detect_sleeper_activation` on each.

### 4. Note on Standard Generation Trials

The current code initializes `activations["standard"]` and `totals["standard"]` (lines 133-135) but never runs standard generation trials (loop at lines 145-164 only runs LDA). Consider removing these counters or adding a standard generation baseline for comparison.

## Expected Speedup

With batch size 8 and fixed-length generation:
- Forward passes reduced from `N * max_tokens * 2` to `(N/8) * max_tokens * 2`
- Practical speedup: 3-5x wall-clock time

## Memory Considerations

Each additional batch item requires ~250-350MB for KV cache and activations. On an A40 (46GB), batch size 8-16 should be safe with two 7B models loaded.

Monitor with `nvidia-smi` during initial testing.

## Implementation Order

Implement changes in this order to minimize debugging complexity:

1. **Phase 1**: Update `_apply_top_p_filtering` and `_sample_token` to handle both 1D and 2D tensors (maintain backward compatibility)
2. **Phase 2**: Update `_lda_step` to accept and use attention masks (test with single-prompt first)
3. **Phase 3**: Implement `generate_lda_batched` method (start with batch_size=2 for testing)
4. **Phase 4**: Add `/generate_lda_batch` endpoint to `server.py`
5. **Phase 5**: Update `run_sleeper_experiment.py` to use batched generation

This phased approach allows testing at each stage before adding complexity.

## Testing Strategy

### 1. Unit Test the Batched Sampling

Test `_sample_token` with a 2D tensor to verify it returns the correct shape:

```python
logits = torch.randn(4, 32000)  # batch_size=4, vocab_size=32000
tokens = _sample_token(logits, temperature=1.0, top_p=1.0)
assert tokens.shape == (4,)
```

### 2. Verify Attention Mask Handling

Compare single-prompt vs batched generation with the same prompt duplicated. Results should be identical (accounting for sampling randomness by setting a seed).

### 3. Check Padding Correctness

Print the tokenized batch to ensure:
- Padding is on the left
- Attention mask is 0 for padding, 1 for real tokens
- All sequences have the same length

### 4. Monitor Memory Usage

Start with batch_size=2, then gradually increase while monitoring GPU memory with `nvidia-smi dmon -s mu`.
