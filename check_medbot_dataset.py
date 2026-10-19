
from pathlib import Path
import json
from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "medbot_phase1_output"

TRAIN_FILE = DATA_DIR / "medquad_train.jsonl"
VALIDATION_FILE = DATA_DIR / "medquad_validation.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 768


def inspect_dataset(file_path, tokenizer):
    total = 0
    empty_answers = 0
    long_examples = 0
    token_lengths = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            question = record["instruction"].strip()
            extra_input = record.get("input", "").strip()
            answer = record["output"].strip()

            if not answer:
                empty_answers += 1
                continue

            if extra_input:
                question += "\nAdditional information: " + extra_input

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            # First create the full conversation as text.
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Then tokenize the text and count the actual token IDs.
            token_ids = tokenizer(
                conversation,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            token_count = len(token_ids)
            total += 1
            token_lengths.append(token_count)

            if token_count > MAX_LENGTH:
                long_examples += 1

            if total == 1:
                print("\nFirst record diagnostic:")
                print("Question:", question[:150])
                print("Answer:", answer[:150])
                print("Formatted text preview:", repr(conversation[:250]))
                print("Actual token count:", token_count)

    token_lengths.sort()

    def percentile(values, p):
        if not values:
            return 0
        index = int((len(values) - 1) * p)
        return values[index]

    print(f"\nFile: {file_path.name}")
    print(f"Usable records: {total:,}")
    print(f"Empty answers skipped: {empty_answers:,}")
    print(f"Examples exceeding {MAX_LENGTH} tokens: {long_examples:,}")

    if token_lengths:
        print("Minimum tokens:", token_lengths[0])
        print("Median total tokens:", percentile(token_lengths, 0.50))
        print("90th percentile tokens:", percentile(token_lengths, 0.90))
        print("Maximum tokens:", token_lengths[-1])


def main():
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    inspect_dataset(TRAIN_FILE, tokenizer)
    inspect_dataset(VALIDATION_FILE, tokenizer)

    print("\nDataset length inspection completed.")


if __name__ == "__main__":
    main()
