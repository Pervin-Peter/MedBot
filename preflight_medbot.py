
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "medbot_phase1_output"

TRAIN_FILE = DATA_DIR / "medquad_train.jsonl"
VALIDATION_FILE = DATA_DIR / "medquad_validation.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 768


def check_file(path):
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not all(key in record for key in ("instruction", "input", "output")):
                raise ValueError(f"Missing fields at line {line_number}")
            count += 1
    return count


def main():
    print("MedBot Full-Training Preflight")
    print("-" * 40)

    for path in (TRAIN_FILE, VALIDATION_FILE):
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"{path.name}: {check_file(path):,} valid records")

    print("\nPyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with TRAIN_FILE.open("r", encoding="utf-8") as file:
        first_record = json.loads(next(line for line in file if line.strip()))

    question = first_record["instruction"].strip()
    answer = first_record["output"].strip()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    conversation = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(conversation, add_special_tokens=False)["input_ids"]

    common_length = 0
    for p_token, f_token in zip(prompt_ids, full_ids):
        if p_token != f_token:
            break
        common_length += 1

    answer_tokens = max(0, min(len(full_ids), MAX_LENGTH) - common_length)

    print("First example total tokens:", len(full_ids))
    print("Prompt tokens masked:", common_length)
    print("Answer tokens available at 768-token limit:", answer_tokens)

    if answer_tokens <= 0:
        raise ValueError("Preflight failed: no answer tokens remain.")

    print("\nPreflight passed. No model training was started.")


if __name__ == "__main__":
    main()
