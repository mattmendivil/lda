# Model Reference

This document lists helpful model pairs for running LDA experiments. Update the `MODEL_ID` and `MODEL_ID_2` variables in `server.py` to switch between different model pairs.

## Model Pairs

### Tiny GPT-2 (for local testing)
- **MODEL_ID**: `sshleifer/tiny-gpt2`
- **MODEL_ID_2**: `sshleifer/tiny-gpt2`

### OLMo 2 1B (Instruct vs Base)
- **MODEL_ID**: `allenai/OLMo-2-0425-1B-Instruct` (post-trained)
- **MODEL_ID_2**: `allenai/OLMo-2-0425-1B` (base)

### OLMo 2 7B (Instruct vs Base)
- **MODEL_ID**: `allenai/OLMo-2-1124-7B-Instruct` (post-trained)
- **MODEL_ID_2**: `allenai/OLMo-2-1124-7B` (base)

### OLMo 2 13B (Instruct vs Base)
- **MODEL_ID**: `allenai/OLMo-2-1124-13B-Instruct` (post-trained)
- **MODEL_ID_2**: `allenai/OLMo-2-1124-13B` (base)

## Notes

- For LDA, `MODEL_ID` is the "after" model (typically post-trained/instruct)
- `MODEL_ID_2` is the "before" model (typically base/pre-trained)
- The alpha parameter in the `/generate_lda` endpoint controls the amplification of differences between models
