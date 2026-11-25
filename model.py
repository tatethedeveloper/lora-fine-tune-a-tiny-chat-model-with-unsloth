"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module

    import torch.nn as nn 
    total = 0
    for param in model.parameters(): 
        total += param.numel()
    return total

# Step 3 - is_model_4bit_quantized
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance

    import torch.nn as nn
    import bitsandbytes as bnb

    return any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return target_modules

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules

    from unsloth import FastLanguageModel

    if target_modules is None:
        target_modules = get_lora_target_modules()

    model = FastLanguageModel.get_peft_model(
        model=model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none"
    )

    return(model.get_base_model())

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    
    import torch.nn as nn 
    total = 0
    for param in model.parameters():
        if param.requires_grad is True:
            total += param.numel()

    return total

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    if total_count == 0:
        return 0.0
    return trainable_count / total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of instruction/response examples for SFT."""
    
    dicts = [
        {
            "instruction": "Count to 10.",
            "response": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
        },
        {
            "instruction": "What is 3 + 4?",
            "response": "7",
        },
        {
            "instruction": "Rewrite 'the dog ran' in uppercase.",
            "response": "THE DOG RAN",
        },
    ]

    return dicts

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string
    
    instruction = example['instruction']
    response = example['response']
    
    formatted_instruction = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    return formatted_instruction

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    
    formatted_instructions = []
    
    for example in examples:
        formatted_instructions.append(format_instruction_example(example))
    
    return formatted_instructions

# Step 12 - build_text_dataset
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings

    from datasets import Dataset

    ds = Dataset.from_dict({"text": list(texts)})

    return ds

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    
    return(tokenizer(text)['input_ids'])

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments (not yet solved)
# TODO: implement

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

