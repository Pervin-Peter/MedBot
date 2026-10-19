
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "medbot_phase1_output"

TRAIN_FILE = DATA_DIR / "medquad_train.jsonl"
VALIDATION_FILE = DATA_DIR / "medquad_validation.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Keep True for the first test. Do not switch yet.
TEST_MODE = True

TEST_SAMPLES = 8
MAX_LENGTH = 768

OUTPUT_DIR = PROJECT_DIR / "medbot_phase2_masking_test"


class MedBotDataset(Dataset):
    """Prepare examples with loss calculated only on answer tokens."""

    def __init__(self, records, tokenizer):
        self.examples = []

        for record in records:
            question = record["instruction"].strip()
            extra_input = record.get("input", "").strip()
            answer = record["output"].strip()

            if extra_input:
                question += "\nAdditional information: " + extra_input

            if not question or not answer:
                continue

            messages = [
                {"role": "user", "content": question}
            ]

                        
            messages = [
                {"role": "user", "content": question}
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_messages = messages + [
                {"role": "assistant", "content": answer}
            ]

            full_text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize the complete conversation and prompt separately.
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_LENGTH,
            )["input_ids"]

            # Find the actual shared prefix instead of assuming the
            # prompt token count matches the complete conversation.
            common_length = 0
            for prompt_token, full_token in zip(prompt_ids, full_ids):
                if prompt_token != full_token:
                    break
                common_length += 1

            labels = list(full_ids)
            labels[:common_length] = [-100] * common_length

            answer_token_count = sum(
                token != -100 for token in labels
            )

            self.examples.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            })


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class AnswerOnlyCollator:
    """Pad examples without changing the answer-only labels."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in examples
            ],
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        padded_labels = []

        for item in examples:
            labels = item["labels"]
            padding_length = max_length - len(labels)
            padded_labels.append(
                labels + [-100] * padding_length
            )

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch


def load_records(file_path, limit=None):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

            if limit is not None and len(records) >= limit:
                break

    return records


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    if not TRAIN_FILE.exists() or not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            "Training or validation dataset file is missing."
        )

    print("MedBot - Answer-Only QLoRA Training")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Test mode:", TEST_MODE)

    train_limit = TEST_SAMPLES if TEST_MODE else None
    validation_limit = TEST_SAMPLES if TEST_MODE else None

    train_records = load_records(TRAIN_FILE, train_limit)
    validation_records = load_records(
        VALIDATION_FILE, validation_limit
    )

    print("Training records loaded:", len(train_records))
    print("Validation records loaded:", len(validation_records))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = MedBotDataset(train_records, tokenizer)
    validation_dataset = MedBotDataset(
        validation_records, tokenizer
    )

    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training or validation dataset is empty.")

    # Confirm that at least one answer token contributes to loss.
    first_labels = train_dataset[0]["labels"]
    answer_tokens = sum(token != -100 for token in first_labels)

    print("Prepared training examples:", len(train_dataset))
    print("Prepared validation examples:", len(validation_dataset))
    print("Answer tokens in first example:", answer_tokens)

    if answer_tokens == 0:
        raise ValueError(
            "No answer tokens remain after truncation. "
            "Increase MAX_LENGTH or inspect the dataset."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    steps = 2 if TEST_MODE else -1

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        max_steps=steps,
        num_train_epochs=1 if not TEST_MODE else 1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        eval_strategy="steps" if TEST_MODE else "epoch",
        eval_steps=2 if TEST_MODE else None,
        save_strategy="no",
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=AnswerOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print("\nTraining finished.")
    print("Training metrics:", train_result.metrics)

    print("\nEvaluating on validation examples...")
    evaluation_metrics = trainer.evaluate()
    print("Evaluation metrics:", evaluation_metrics)

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    results = {
        "model": MODEL_NAME,
        "test_mode": TEST_MODE,
        "training_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "max_length": MAX_LENGTH,
        "train_metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
    }

    with (OUTPUT_DIR / "training_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2, default=str)

    print("\nSUCCESS: Answer-only training test completed.")
    print("Adapter:", adapter_dir)
    print("Results:", OUTPUT_DIR / "training_results.json")
    print("This is still a small technical test, not a final model.")

    del trainer, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
