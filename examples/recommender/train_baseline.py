"""Baseline GPT-2 fine-tuning loop to try the recommender on (example; never executed by Compute Cast).

Deliberately simple: FP32 everywhere, one large microbatch, no gradient accumulation, no activation
checkpointing, a single GPU and a plain DataLoader. The recommender can suggest batch/accumulation
tuning, BF16 autocast, checkpointing, data-loader settings or multi-GPU strategies for it.
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

MODEL_ID = "openai-community/gpt2"
SEQUENCE_LENGTH = 1024
MICROBATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_STEPS = 2000


def make_loader(dataset):
    # dataset yields dicts with fixed-length "input_ids" (already tokenized and padded)
    return DataLoader(dataset, batch_size=MICROBATCH_SIZE, shuffle=True, num_workers=0)


def train(dataset):
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    step = 0
    for batch in make_loader(dataset):
        input_ids = batch["input_ids"].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {outputs.loss.item():.4f}")
        if step >= MAX_STEPS:
            break
    model.save_pretrained("checkpoints/final")
