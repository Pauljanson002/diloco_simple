from transformers import AutoTokenizer, FlaxLlamaForCausalLM, LlamaConfig

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1", use_fast=True
)
tokenizer.pad_token = "</s>"
config_path = "/home/paul/workspace/scaling_l2o/extern/diloco_simple/config_14m.json"
config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=config_path)
model = FlaxLlamaForCausalLM(config)

inputs = tokenizer("Hello, my dog is ", return_tensors="np")
outputs = model(**inputs)

# retrieve logts for next token
next_token_logits = outputs.logits[:, -1]

# print the next word
# Get the predicted token ID (the one with highest probability)
next_token_id = next_token_logits.argmax(-1)[0]

# Convert the token ID back to a word
next_word = tokenizer.decode(next_token_id)

print(f"Next predicted word: {next_word}")
