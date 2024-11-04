import torch

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

device = ('cude' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')

model_name = 'nvidia/Mistral-NeMo-Minitron-8B-Base'
# RuntimeError: MPS backend out of memory (MPS allocated: 8.99 GB, other allocations: 36.06 MB, max allowed: 9.07 GB).
# Tried to allocate 180.00 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for
# memory allocations (may cause system failure).

model_name = 'microsoft/phi-2'
# RuntimeError: MPS backend out of memory (MPS allocated: 8.79 GB, other allocations: 213.32 MB, max allowed: 9.07 GB).
# Tried to allocate 100.00 MB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for
# memory allocations (may cause system failure).

token = None

tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)
pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if device == 'cuda' else -1)


