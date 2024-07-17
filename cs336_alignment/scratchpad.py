from huggingface_hub import HfFolder
from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B"  # or the specific model you want to check
HfFolder.save_token("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")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(f"Successfully loaded {model_name}. You have access.")
except Exception as e:
    print(f"Error loading {model_name}. You might not have access: {str(e)}")