import torch

# Removed wandb import
from cyclopts import App
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)

app = App()


@app.default
def main(
    batch_size: int = 512,
    per_device_batch_size: int = 32,
    seq_length: int = 1024,
    warmup_steps: int = 1000,
    total_steps: int = 88_000,
    config_path: str = "config_14m.json",  # Removed project parameter
    lr: float = 4e-4,
):
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = batch_size // per_device_batch_size

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Removed wandb initialization

    # Load model configuration
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=config_path)
    model = LlamaForCausalLM(config).to(device)
    print(
        f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=0.1, lr=lr, betas=(0.9, 0.95)
    )

    # Setup learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1", use_fast=True
    )
    tokenizer.pad_token = "</s>"  # Ensure pad token is set

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("PrimeIntellect/c4-tiny", "en", verification_mode="no_checks")

    # Tokenize dataset
    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=seq_length)
        return outputs

    print("Tokenizing dataset...")
    tokenized_datasets = ds.map(
        tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]
    )

    # Setup data collator and dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        collate_fn=data_collator,
        batch_size=per_device_batch_size,
        shuffle=True,
    )

    print(f"Starting training with effective batch size of {batch_size}")

    # Training loop
    model.train()
    global_step = 0
    loss_batch = 0

    for batch_idx, batch in enumerate(train_dataloader):
        # Calculate position within gradient accumulation
        step_within_grad_acc = (batch_idx + 1) % gradient_accumulation_steps

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = (
            outputs.loss / gradient_accumulation_steps
        )  # Scale loss for gradient accumulation

        # Accumulate loss for logging
        loss_batch += loss.detach()

        # Backward pass
        loss.backward()

        # Update weights after accumulating gradients
        if step_within_grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Simple console logging (no wandb)
            current_lr = scheduler.get_last_lr()[0]
            perplexity = torch.exp(loss_batch).item()

            print(
                f"Step: {global_step}, Loss: {loss_batch.item():.4f}, Perplexity: {perplexity:.2f}, LR: {current_lr:.6f}"
            )

            loss_batch = 0

            # Removed checkpoint saving

            # Check if we've reached total_steps
            if global_step >= total_steps:
                break

    print("Training completed.")

    # Removed final model saving


if __name__ == "__main__":
    app()
