
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

PROJECT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = PROJECT_DIR / "medbot_phase2_full" / "lora_adapter"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Adapter folder not found: {ADAPTER_DIR}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print("Loading base model in 4-bit mode...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )

    print("Loading full-training LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    question = "What is relapsing polychondritis?"
    messages = [{"role": "user", "content": question}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device)

    print("\nQuestion:", question)
    print("\nGenerating answer...")

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_tokens = output[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    print("\nMedBot answer:")
    print(answer)

    print("\nAdapter loading and generation test completed.")


if __name__ == "__main__":
    main()
