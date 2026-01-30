"""
FastAPI server for Logit Diff Amplification (LDA).

Provides HTTP endpoints for generating text with LDA and comparing
post-trained vs pre-trained model outputs.
"""

import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lda import LDAModelPair

# loads env variables. helpful for local dev
load_dotenv()

# Model configuration
MODEL_AFTER_ID = os.environ.get("MODEL_AFTER_ID", "yifever/sleeper-agent")
MODEL_BEFORE_ID = os.environ.get("MODEL_BEFORE_ID", "decapoda-research/llama-7b-hf")
DEVICE = os.environ.get("DEVICE")  # None means auto-detect in LDAModelPair

app = FastAPI(
    title="Logit Diff Amplification API",
    description="API for surfacing rare behaviors in post-trained models using LDA",
    version="1.0.0"
)

_model_pair: LDAModelPair | None = None
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
    alpha: float = Field(1.0, ge=0.0, le=10.0, description="Amplification factor for logit differences")
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
    """Initialize the LDA model pair on server startup."""
    global _model_pair
    
    print(f"Initializing LDA with models:", flush=True)
    print(f"  After:  {MODEL_AFTER_ID}", flush=True)
    print(f"  Before: {MODEL_BEFORE_ID}", flush=True)
    
    _model_pair = LDAModelPair(
        model_after_id=MODEL_AFTER_ID,
        model_before_id=MODEL_BEFORE_ID,
        device=DEVICE
    )
    
    print("\nServer ready!", flush=True)


@app.get("/tokenizer_compatibility", response_model=TokenizerCompatibilityInfo)
def tokenizer_compatibility() -> TokenizerCompatibilityInfo:
    """Check if the two tokenizers are compatible for LDA."""
    if _model_pair is None:
        raise HTTPException(status_code=503, detail="Models not initialized yet")
    
    compat = _model_pair.get_tokenizer_compatibility()
    
    return TokenizerCompatibilityInfo(
        compatible=compat.compatible,
        vocab_size_1=compat.vocab_size_after,
        vocab_size_2=compat.vocab_size_before,
        vocabs_match=compat.vocabs_match,
        warning=compat.warning,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text from both models (after and before) for comparison."""
    if _model_pair is None:
        raise HTTPException(status_code=503, detail="Models not initialized yet")

    with _lock:
        # Generate with after model
        result_after = _model_pair.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.do_sample,
            apply_chat_template=req.apply_chat_template,
            system_prompt=req.system_prompt,
            use_after_model=True
        )
        
        # Generate with before model
        result_before = _model_pair.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.do_sample,
            apply_chat_template=req.apply_chat_template,
            system_prompt=req.system_prompt,
            use_after_model=False
        )

    return GenerateResponse(
        prompt=req.prompt,
        model_1=ModelResponse(
            model_id=MODEL_AFTER_ID,
            completion=result_after.completion,
            text=result_after.text,
        ),
        model_2=ModelResponse(
            model_id=MODEL_BEFORE_ID,
            completion=result_before.completion,
            text=result_before.text,
        ),
    )


@app.post("/generate_lda", response_model=LDAGenerateResponse)
def generate_lda(req: LDAGenerateRequest) -> LDAGenerateResponse:
    """Generate text using Logit Diff Amplification (LDA)."""
    if _model_pair is None:
        raise HTTPException(status_code=503, detail="Models not initialized yet")

    with _lock:
        result = _model_pair.generate_lda(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            alpha=req.alpha,
            apply_chat_template=req.apply_chat_template,
            system_prompt=req.system_prompt
        )

    # Reconstruct formatted prompt for response
    from lda import _format_as_chat
    formatted_prompt = req.prompt
    if req.apply_chat_template:
        formatted_prompt = _format_as_chat(req.prompt, _model_pair.tokenizer_after, req.system_prompt)

    return LDAGenerateResponse(
        prompt=req.prompt,
        formatted_prompt=formatted_prompt,
        completion=result.completion,
        text=result.text,
        alpha=req.alpha,
        model_after=MODEL_AFTER_ID,
        model_before=MODEL_BEFORE_ID,
        tokens_generated=result.tokens_generated,
        stopped_early=result.stopped_early,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, log_level="info")
