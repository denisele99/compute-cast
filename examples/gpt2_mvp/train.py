"""Example GPT-2 fine-tuning script (illustrative). Compute Cast only parses it statically."""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


def tokenize(batch):
    return tokenizer(batch["text"], max_length=MAX_LENGTH, padding="max_length", truncation=True)


dataset = load_dataset("text", data_files={"train": "train.txt"})["train"].map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="out/gpt2-example",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    bf16=True,
    optim="adamw_torch",
)
Trainer(model=model, args=args, train_dataset=dataset).train()
