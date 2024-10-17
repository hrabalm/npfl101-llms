from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch

DATASET_PATH = "/storage/brno12-cerit/home/hrabalm/datasets/npfl101_test_dataset"  # TODO: dataset name

dataset = load_from_disk(DATASET_PATH)
# dataset = load_dataset("username/datasetname", split="train")  # you can also download the dataset from hub instead

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=["all-linear"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

sft_config = SFTConfig(
    output_dir="/tmp"
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=peft_config,
)

trainer.train()

model.save_to_disk("/storage/brno12-cerit/home/hrabalm/models/npfl101_test_model")
# you can also push the model to hub, see docs
