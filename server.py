import os
import threading
import warnings

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# loads env variables. helpful for local dev
load_dotenv()

"""
NOTE:
MODEL_ID (primary) must be the post-trained Instruct model, since its tokenizer and
chat template define the canonical prompt rendering for LDA. MODEL_ID_2 (secondary)
must be the pre-trained/base model, which is evaluated on the same token sequence
so that amplified logits capture only post-training weight changes.# Primary model configuration
"""
MODEL_ID = "allenai/OLMo-2-0425-1B-Instruct"
# Secondary model configuration
MODEL_ID_2 = "allenai/OLMo-2-0425-1B"
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

_tokenizer = None
_model = None
_tokenizer_2 = None
_model_2 = None
_tokenizers_compatible = False
_lock = threading.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(80, ge=1, le=512)
    do_sample: bool = True
    temperature: float = Field(0.8, gt=0.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    apply_chat_template: bool = Field(True, description="Whether to apply chat template formatting")
    system_prompt: str | None = Field(None, description="Optional system prompt for chat template")


class LDAGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user message (will be formatted as chat)")
    max_new_tokens: int = Field(80, ge=1, le=512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.8, gt=0.0, description="Temperature for sampling")
    top_p: float = Field(0.95, gt=0.0, le=1.0, description="Top-p (nucleus) sampling threshold")
    alpha: float = Field(1.0, gt=0.0, le=10.0, description="Amplification factor for logit differences")
    do_sample: bool = Field(True, description="Whether to use sampling (always True for LDA)")
    apply_chat_template: bool = Field(True, description="Whether to apply chat template formatting")
    system_prompt: str | None = Field(None, description="Optional system prompt for chat template")


class ModelResponse(BaseModel):
    model_id: str
    completion: str
    text: str


class GenerateResponse(BaseModel):
    prompt: str
    model_1: ModelResponse
    model_2: ModelResponse


class LDAGenerateResponse(BaseModel):
    prompt: str
    formatted_prompt: str
    completion: str
    text: str
    alpha: float
    model_after: str
    model_before: str
    tokens_generated: int
    stopped_early: bool


class TokenizerCompatibilityInfo(BaseModel):
    compatible: bool
    vocab_size_1: int
    vocab_size_2: int
    vocabs_match: bool
    warning: str | None = None


@app.on_event("startup")
def _startup() -> None:
    global _model, _tokenizer, _model_2, _tokenizer_2, _tokenizers_compatible
    
    # Load first model
    print(f"Loading tokenizer 1: {MODEL_ID}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Loading model weights 1: {MODEL_ID}", flush=True)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    _model.to(DEVICE)
    _model.eval()
    print(f"Model 1 ready on {DEVICE}", flush=True)
    
    # Load second model
    print(f"Loading tokenizer 2: {MODEL_ID_2}", flush=True)
    _tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_ID_2)
    print(f"Loading model weights 2: {MODEL_ID_2}", flush=True)
    _model_2 = AutoModelForCausalLM.from_pretrained(MODEL_ID_2)
    _model_2.to(DEVICE)
    _model_2.eval()
    print(f"Model 2 ready on {DEVICE}", flush=True)
    
    # Validate tokenizer compatibility
    print("\nValidating tokenizer compatibility...", flush=True)
    vocab1 = _tokenizer.get_vocab()
    vocab2 = _tokenizer_2.get_vocab()
    
    vocab_size_1 = len(vocab1)
    vocab_size_2 = len(vocab2)
    vocabs_match = vocab1 == vocab2
    
    print(f"  Tokenizer 1 vocab size: {vocab_size_1}", flush=True)
    print(f"  Tokenizer 2 vocab size: {vocab_size_2}", flush=True)
    print(f"  Vocabularies match: {vocabs_match}", flush=True)
    
    if not vocabs_match:
        warnings.warn(
            f"Tokenizers have incompatible vocabularies! "
            f"LDA will likely produce incorrect results. "
            f"Model 1: {vocab_size_1} tokens, Model 2: {vocab_size_2} tokens"
        )
        _tokenizers_compatible = False
        print("  WARNING: Tokenizers are NOT compatible for LDA!", flush=True)
    else:
        _tokenizers_compatible = True
        print("  ✓ Tokenizers are compatible for LDA", flush=True)
    
    print("\nServer ready!", flush=True)


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


def _generate_lda(
    model_after,
    model_before,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    alpha: float
) -> tuple[str, str, int, bool]:
    """
    Generate text using logit diff amplification.
    
    logits_amplified = logits_after + alpha * (logits_after - logits_before)
    
    Returns:
        completion: Generated text without prompt
        text: Full text including prompt
        tokens_generated: Number of tokens actually generated
        stopped_early: Whether generation stopped before max_new_tokens
    """
    # Tokenize initial prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    generated_ids = input_ids.clone()
    
    tokens_generated = 0
    stopped_early = False
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Get logits from both models
            outputs_after = model_after(generated_ids)
            logits_after = outputs_after.logits[0, -1, :]  # Last token logits
            
            outputs_before = model_before(generated_ids)
            logits_before = outputs_before.logits[0, -1, :]  # Last token logits
            
            # Compute amplified logits
            # logits_amplified = logits_after + alpha * (logits_after - logits_before)
            logits_diff = logits_after - logits_before
            logits_amplified = logits_after + alpha * logits_diff
            
            # Sample next token from amplified logits
            next_token_id = _sample_token(logits_amplified, temperature, top_p)
            
            tokens_generated += 1
            
            # Check for EOS token
            if next_token_id == tokenizer.eos_token_id:
                stopped_early = True
                break
            
            # Append to generated sequence
            next_token_tensor = torch.tensor([[next_token_id]], device=DEVICE)
            generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
    
    # Decode results
    input_len = input_ids.shape[1]
    completion_ids = generated_ids[0, input_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return completion, text, tokens_generated, stopped_early


def _generate_with_model(model, tokenizer, prompt: str, req: GenerateRequest) -> tuple[str, str]:
    """Helper function to generate text with a given model and tokenizer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0, input_len:]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return completion, text


@app.get("/tokenizer_compatibility", response_model=TokenizerCompatibilityInfo)
def tokenizer_compatibility() -> TokenizerCompatibilityInfo:
    """Check if the two tokenizers are compatible for LDA."""
    if _tokenizer is None or _tokenizer_2 is None:
        raise HTTPException(status_code=503, detail="Models not initialized yet")
    
    vocab1 = _tokenizer.get_vocab()
    vocab2 = _tokenizer_2.get_vocab()
    
    vocab_size_1 = len(vocab1)
    vocab_size_2 = len(vocab2)
    vocabs_match = vocab1 == vocab2
    
    warning = None
    if not vocabs_match:
        warning = (
            "Tokenizers have incompatible vocabularies. "
            "LDA may produce incorrect or nonsensical results. "
            "For best results, use models with identical tokenizers."
        )
    
    return TokenizerCompatibilityInfo(
        compatible=vocabs_match,
        vocab_size_1=vocab_size_1,
        vocab_size_2=vocab_size_2,
        vocabs_match=vocabs_match,
        warning=warning,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if _model is None or _tokenizer is None or _model_2 is None or _tokenizer_2 is None:
        raise RuntimeError("Models not initialized yet")

    # Apply chat template if requested
    prompt_1 = req.prompt
    prompt_2 = req.prompt
    
    if req.apply_chat_template:
        prompt_1 = _format_as_chat(req.prompt, _tokenizer_2, req.system_prompt)
        prompt_2 = _format_as_chat(req.prompt, _tokenizer_2, req.system_prompt)

    with _lock:
        # Generate with first model
        completion_1, text_1 = _generate_with_model(_model, _tokenizer, prompt_1, req)
        
        # Generate with second model
        completion_2, text_2 = _generate_with_model(_model_2, _tokenizer_2, prompt_2, req)

    return GenerateResponse(
        prompt=req.prompt,
        model_1=ModelResponse(
            model_id=MODEL_ID,
            completion=completion_1,
            text=text_1,
        ),
        model_2=ModelResponse(
            model_id=MODEL_ID_2,
            completion=completion_2,
            text=text_2,
        ),
    )


@app.post("/generate_lda", response_model=LDAGenerateResponse)
def generate_lda(req: LDAGenerateRequest) -> LDAGenerateResponse:
    if _model is None or _tokenizer is None or _model_2 is None or _tokenizer_2 is None:
        raise HTTPException(status_code=503, detail="Models not initialized yet")
    
    # Warn if tokenizers are incompatible
    if not _tokenizers_compatible:
        warnings.warn(
            "Tokenizers are incompatible! LDA results may be nonsensical. "
            "Check /tokenizer_compatibility endpoint for details."
        )

    # Apply chat template if requested
    formatted_prompt = req.prompt
    if req.apply_chat_template:
        formatted_prompt = _format_as_chat(req.prompt, _tokenizer, req.system_prompt)

    with _lock:
        completion, text, tokens_generated, stopped_early = _generate_lda(
            model_after=_model,
            model_before=_model_2,
            tokenizer=_tokenizer,
            prompt=formatted_prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            alpha=req.alpha
        )

    return LDAGenerateResponse(
        prompt=req.prompt,
        formatted_prompt=formatted_prompt,
        completion=completion,
        text=text,
        alpha=req.alpha,
        model_after=MODEL_ID,
        model_before=MODEL_ID_2,
        tokens_generated=tokens_generated,
        stopped_early=stopped_early,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, log_level="info")
