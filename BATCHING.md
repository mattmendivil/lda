# Implementing Batched LDA Generation

This document describes how to add batching support to speed up LDA experiments.

## Overview

The current implementation processes one prompt at a time. Batching allows multiple prompts to be processed in a single forward pass, significantly improving GPU utilization and throughput.

## Prerequisites

- Fixed-length generation (always generate `max_new_tokens`, ignore EOS)
- All prompts in a batch padded/truncated to the same input length

## Changes to `lda.py`

### 1. Modify `_lda_step` (lines 237-283)

Current behavior: Accepts `input_ids` with shape `[1, seq_len]` and returns a single token.

New behavior: Accept `input_ids` with shape `[batch_size, seq_len]` and return `batch_size` tokens.

Key changes:
- Line 266: Change `logits[0, -1, :]` to `logits[:, -1, :]` to preserve batch dimension
- Line 273: Same change for the before model
- Line 281: `_sample_token` must handle batched logits

### 2. Modify `_sample_token` (lines 92-117)

Current behavior: Samples one token from a 1D logits tensor.

New behavior: Sample N tokens from a 2D `[batch_size, vocab_size]` tensor.

Use `torch.multinomial` on each row, or vectorize with batch-aware sampling.

### 3. Add `generate_lda_batched` method

Create a new method that accepts a list of prompts instead of a single prompt.

Structure:
1. Tokenize all prompts, pad to same length
2. Stack into a single tensor of shape `[batch_size, max_input_len]`
3. Initialize KV caches as None
4. Loop for exactly `max_new_tokens` iterations (no EOS checking)
5. Call batched `_lda_step` each iteration
6. Concatenate returned tokens to the batch
7. Decode all sequences and return list of results

Reference the existing `generate_lda` (lines 285-369) for the single-prompt structure.

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

## Changes to `run_experiment.py`

### 1. Batch Prompt Loading

Instead of iterating one prompt at a time, collect prompts into batches of size 8-16.

### 2. Single Request Per Batch

Send one HTTP request containing all prompts in the batch, rather than N individual requests.

### 3. Process Results

The response will contain a list of completions. Run `detect_sleeper_activation` on each.

## Expected Speedup

With batch size 8 and fixed-length generation:
- Forward passes reduced from `N * max_tokens * 2` to `(N/8) * max_tokens * 2`
- Practical speedup: 3-5x wall-clock time

## Memory Considerations

Each additional batch item requires ~250-350MB for KV cache and activations. On an A40 (46GB), batch size 8-16 should be safe with two 7B models loaded.

Monitor with `nvidia-smi` during initial testing.
