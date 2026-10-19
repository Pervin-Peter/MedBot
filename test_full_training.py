
from pathlib import Path
import json
import math
import sys
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
MAX_LENGTH = 768

OUTPUT_DIR = PROJECT_DIR / "medbot_phase2_smoke_test"


class MedBotDataset(Dataset):
    """Create causal-LM examples with answer-only loss."""

    def __init__(self, records, tokenizer):
        self.examples = []
        self.truncated_examples = 0
        self.skipped_examples = 0

        for record in records:
            question = record["instruction"].strip()
            extra_input = record.get("input", "").strip()
            answer = record["output"].strip()

            if extra_input:
                question += "\nAdditional information: " + extra_input

            if not question or not answer:
                self.skipped_examples += 1
                continue

            prompt_messages = [
                {"role": "user", "content": question}
            ]

            full_messages = prompt_messages + [
                {"role": "assistant", "content": answer}
            ]

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
            )["input_ids"]

            if len(full_ids) > MAX_LENGTH:
                self.truncated_examples += 1
                full_ids = full_ids[:MAX_LENGTH]

            # Find the exact common prefix to avoid token-boundary
            # mismatches between the prompt and complete conversation.
            common_length = 0
            for prompt_token, full_token in zip(prompt_ids, full_ids):
                if prompt_token != full_token:
                    break
                common_length += 1

            labels = list(full_ids)
            labels[:common_length] = [-100] * common_length

            if not any(token != -100 for token in labels):
                self.skipped_examples += 1
                continue

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
    """Pad examples while preserving answer-only labels."""

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

        batch_length = batch["input_ids"].shape[1]
        labels = []

        for item in examples:
            row = item["labels"]
            row = row + [-100] * (batch_length - len(row))
            labels.append(row)

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def load_records(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
        

    if "--preflight" in sys.argv:
        print("Preflight mode: checking configuration and dataset preparation.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    if not TRAIN_FILE.exists() or not VALIDATION_FILE.exists():
        raise FileNotFoundError("Training or validation file is missing.")

    print("MedBot - Full QLoRA Training")
    print("Model:", MODEL_NAME)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Sequence limit:", MAX_LENGTH)

    print("\nLoading dataset records...")
    train_records = load_records(TRAIN_FILE)[:8]
    validation_records = load_records(VALIDATION_FILE)[:8]

    print("Training records:", len(train_records))
    print("Validation records:", len(validation_records))

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit mode...")
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

    print("\nTokenizing training data...")
    train_dataset = MedBotDataset(train_records, tokenizer)

    print("Tokenizing validation data...")
    validation_dataset = MedBotDataset(validation_records, tokenizer)

    print("\nTraining examples prepared:", len(train_dataset))
    print("Training examples truncated:", train_dataset.truncated_examples)
    print("Training examples skipped:", train_dataset.skipped_examples)
    print("Validation examples prepared:", len(validation_dataset))
    print("Validation examples truncated:", validation_dataset.truncated_examples)
    print("Validation examples skipped:", validation_dataset.skipped_examples)

    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Prepared training or validation dataset is empty.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Approximately one epoch; keep batch size at 1 for the 8 GB GPU.
    steps_per_epoch = math.ceil(len(train_dataset))
    print("Approximate optimizer steps per epoch:", steps_per_epoch)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=1,
        max_steps=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
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

    print("\nStarting full training.")
    print("This may take a long time. Keep the laptop plugged in.")
    train_result = trainer.train()

    print("\nEvaluating on the full validation set...")
    evaluation_metrics = trainer.evaluate()

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    results = {
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "training_records_loaded": len(train_records),
        "training_examples_prepared": len(train_dataset),
        "training_examples_truncated": train_dataset.truncated_examples,
        "training_examples_skipped": train_dataset.skipped_examples,
        "validation_records_loaded": len(validation_records),
        "validation_examples_prepared": len(validation_dataset),
        "validation_examples_truncated": validation_dataset.truncated_examples,
        "validation_examples_skipped": validation_dataset.skipped_examples,
        "training_metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
    }

    with (OUTPUT_DIR / "training_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2, default=str)

    print("\nFULL TRAINING COMPLETED")
    print("Adapter:", adapter_dir)
    print("Results:", OUTPUT_DIR / "training_results.json")

    del trainer, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
