import torch
from transformers import AutoModelForCausalLM

MODEL = "/storage/brno12-cerit/home/hrabalm/models/npfl101_test_model"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
merged_model = model.merge_and_unload(progressbar=True)
model.save_pretrained(MODEL + "_bf16")
