"""
Example usage of the LDA library without the API server.

This demonstrates how to use the LDAModelPair class directly in your code.
"""

from lda import LDAModelPair


def main():
    print("=" * 80)
    print("LDA Example: Direct Library Usage")
    print("=" * 80)
    
    # Initialize model pair
    print("\n1. Initializing models...")
    model_pair = LDAModelPair(
        model_after_id="allenai/OLMo-2-0425-1B-Instruct",
        model_before_id="allenai/OLMo-2-0425-1B"
    )
    
    # Check tokenizer compatibility
    print("\n2. Checking tokenizer compatibility...")
    compat = model_pair.get_tokenizer_compatibility()
    if compat.compatible:
        print(f"   ✓ Tokenizers are compatible!")
    else:
        print(f"   ⚠ Warning: {compat.warning}")
    
    # Example prompt
    prompt = "What should I do if I feel bored?"
    
    # Standard generation from after model
    print(f"\n3. Standard generation (after model)...")
    print(f"   Prompt: {prompt}")
    result_after = model_pair.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.95,
        use_after_model=True
    )
    print(f"   Response: {result_after.completion}")
    
    # Standard generation from before model
    print(f"\n4. Standard generation (before model)...")
    print(f"   Prompt: {prompt}")
    result_before = model_pair.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.95,
        use_after_model=False
    )
    print(f"   Response: {result_before.completion}")
    
    # LDA generation with different alpha values
    print(f"\n5. LDA generation with α=0.5...")
    print(f"   Prompt: {prompt}")
    result_lda = model_pair.generate_lda(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.95,
        alpha=0.5
    )
    print(f"   Response: {result_lda.completion}")
    print(f"   Tokens generated: {result_lda.tokens_generated}")
    print(f"   Stopped early: {result_lda.stopped_early}")
    
    print(f"\n6. LDA generation with α=2.0...")
    print(f"   Prompt: {prompt}")
    result_lda_high = model_pair.generate_lda(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.95,
        alpha=2.0
    )
    print(f"   Response: {result_lda_high.completion}")
    print(f"   Tokens generated: {result_lda_high.tokens_generated}")
    print(f"   Stopped early: {result_lda_high.stopped_early}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
