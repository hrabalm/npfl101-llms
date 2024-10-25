from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch

DATASET_PATH = "/storage/brno12-cerit/home/hrabalm/datasets/npfl101_test_dataset"  # TODO: change dataset name

MODEL = "utter-project/EuroLLM-1.7B"
# If you have succesfully setup your Huggingface account and access tokens, you
# can also use some gated model you have been given access to, for example
# Mistral-7b-v0.3, Gemma-2-2b or Llama3.1.
# You should check their pages on HF first and check that you have access to
# to them.
# https://huggingface.co/google/gemma-2-2b
# https://huggingface.co/mistralai/Mistral-7B-v0.3
# https://huggingface.co/meta-llama/Llama-3.1-8B

# MODEL = "google/gemma-2-2b"

# Note that if you use larger models, you might have to check adjust the batch
# size so that the models fit the the GPU VRAM you specify in qsub_*.sh files.

OUTPUT_DIRECTORY = "/storage/brno12-cerit/home/hrabalm/models/npfl101_test_model"  # TODO: change output directory

dataset = load_from_disk(DATASET_PATH)
# dataset = load_dataset("username/datasetname", split="train")  # you can also download the dataset from hub instead

# Our dataset is not in the format that SFTTrainer expects, see
# https://huggingface.co/docs/trl/dataset_formats
# So we need to preprocess our dataset

def preprocess_function(examples):
    # note the token used at the end depends on the model we are training
    EOS = "<eos>"
    return {
        "text": f"Translate the following Czech sentence to English.\nCzech: {examples["source_text"]}\nEnglish: {examples["target_text"]}{EOS}",
    }
dataset = dataset.map(preprocess_function)
# alternative, you can pass formatting_function to SFTTrainer, see https://huggingface.co/docs/trl/main/sft_trainer#format-your-input-prompts

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules="all-linear", # or list of modules
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

sft_config = SFTConfig(
    # max_seq_length=2048,  # you should properly set this depending on your data, task, model and VRAM
    learning_rate=2e-5,
    # fp16=True,
    bf16=True,
    optim="adamw_8bit",  # saves VRAM
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=10,
    logging_steps=1,
    num_train_epochs=2,
    per_device_train_batch_size=1,  # you should try to increase this as much as possible while still avoiding out of memory errors
    # gradient_accumulation_steps=1,  # can be used to simulate larger batch sizes
    output_dir=".",  # where to save the model checkpoints
    # gradient_checkpointing=True,  # save VRAM at the cost of computation time
    seed=42,
    # save_strategy="steps",
    # save_steps=100,
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    args=sft_config,
    peft_config=peft_config,
)

trainer.train()

model.save_pretrained(OUTPUT_DIRECTORY)

# you can also push the model to hub, see docs

# Optional exercises:
# - Currently, we train the model by calculating the loss on all tokens (including the prompt). See https://huggingface.co/docs/trl/main/sft_trainer#train-on-completions-only for an example of how to train only on the completions.
# - Try training on different or larger models
# - Try to disable the quantization if the model is small enough
# - Add evaluation dataset, set eval_steps and eval_strategy
# - Import wandb and log the training to wandb
# - Go through the documentation https://huggingface.co/docs/trl/main/sft_trainer
# - Initialize QLoRA weights using LoftQ, see https://huggingface.co/docs/peft/developer_guides/quantization#loftq-initialization
